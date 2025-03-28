import fitz  # PyMuPDF for PDF extraction
import faiss  # FAISS for vector storage
import numpy as np  # NumPy for array storage & similarity search
import re  # Regex for text processing
import json  # JSON for storing faiss_text_store
import redis  # Redis for caching question-answer pairs
import os  # OS operations
import openai  # OpenAI API client
import pandas as pd  # Pandas for Excel processing
import streamlit as st  # Streamlit for UI
import time  # For timestamps
from sentence_transformers import SentenceTransformer  # Sentence embeddings
from scipy.spatial.distance import cosine  # For similarity search


FAISS_INDEX_FILE = "faiss_index.idx"
TEXT_STORE_FILE = "faiss_text.json"

SYNONYM_MAP = {
    "course": ["program", "course", "degree", "qualification"],
    "program": ["course", "degree", "offering"],
    "degree": ["program", "course", "academic qualification"],
    "mba": ["Master of Business Administration"],
    "bsc": ["Bachelor of Science"],
    "msc": ["Master of Science"],
    "phd": ["Doctorate", "Doctor of Philosophy"],
}

print("Step 1:", time.strftime('%Y-%m-%d %H:%M:%S'))

# Connect to Redis (Ensure Redis is running)
#redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# ✅ Get Redis URL from Streamlit secrets
REDIS_URL = st.secrets["redis"]["url"]

# ✅ Connect to Redis
redis_client = redis.StrictRedis.from_url(REDIS_URL, decode_responses=True)

# Load OpenAI API Client
@st.cache_resource
def get_openai_client():
    return openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# Load Embedding Model (Efficient)
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS Index & `faiss_text_store` Together
# Load FAISS Index (always expect it to be present)
def load_faiss():
    index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"✅ FAISS index loaded from {FAISS_INDEX_FILE}. Contains {index.ntotal} embeddings.")
    return index

# Load FAISS text store (always expect it to be present)
def load_faiss_text_store():
    with open(TEXT_STORE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

index = load_faiss()
faiss_text_store = load_faiss_text_store()
last_faiss_id = index.ntotal

index = load_faiss()
faiss_text_store = load_faiss_text_store()
last_faiss_id = index.ntotal 
    
print("Step 2:",  time.strftime('%Y-%m-%d %H:%M:%S'))


# Ensure FAISS & `faiss_text_store` Stay in Sync
if index.ntotal != len(faiss_text_store):
    print(" FAISS and faiss_text_store are out of sync. Required reload.")
else:
    print(" FAISS and faiss_text_store are in sync.")

# Extract Text from PDF
@st.cache_resource
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

# Check is University Heading
def is_university_heading(text):
    """Checks if a line is a university name."""
    UNIVERSITIES = json.loads(redis_client.get("UNIVERSITIES"))
    return text.strip().lower() in [u.lower() for u in UNIVERSITIES]

# plits text by headings while ensuring university names are kept in all related chunks
def split_text_by_headings(text, chunk_size, overlap):
    """Splits text by headings while ensuring university names are kept in all related chunks.
       Merges small chunks of the same university to avoid losing context, without redundant university name repetition."""
    
    lines = text.split("\n")
    final_sections = []
    current_chunk = ""
    current_university = None  # Use None to avoid false insertions
    
    # Debugging: Print available university names from Redis
    print("UNIVERSITIES", json.loads(redis_client.get("UNIVERSITIES")))

    for line in lines:
        if is_university_heading(line):  #Detect University Heading
            if current_chunk:
                final_sections.append(current_chunk.strip())  # Save previous chunk

            current_university = line.strip()  # Set university heading
            current_chunk = current_university + "\n"  # Start new chunk with university name

        elif re.match(r"\n\s*(?=[A-Z][^\n]+\n={3,})", line):  # Standard heading detection
            if current_chunk:
                if len(current_chunk) + len(line) < chunk_size:
                    current_chunk += "\n"+ line + "\n"
                else:
                    final_sections.append(current_chunk.strip())
                    current_chunk = line.strip() + "\n"

            else:
                current_chunk = line.strip() + "\n"

        else:
            if len(current_chunk) + len(line) < chunk_size:
                current_chunk += line + "\n"
            else:
                final_sections.append(current_chunk.strip())
                current_chunk = line + "\n"  

    if current_chunk:
        final_sections.append(current_chunk.strip())  # Add the last chunk

    # Merging small chunks of the same university
    merged_sections = []
    prev_chunk = ""
    prev_university = None  # Keep track of last university heading

    for section in final_sections:
        lines = section.split("\n")
        university = lines[0] if is_university_heading(lines[0]) else prev_university

        # Extract content without the university name for merging
        section_body = "\n".join(lines[1:]) if is_university_heading(lines[0]) else section

        if prev_chunk and university == prev_university and len(prev_chunk) + len(section_body) < chunk_size:
            prev_chunk += " " + section_body.strip()  # Merge while maintaining sentence structure
        else:
            if prev_chunk:
                merged_sections.append(prev_chunk.strip())  # Save previous chunk
            prev_chunk = section  # Start a new chunk with university name only ONCE
            prev_university = university

    if prev_chunk:
        merged_sections.append(prev_chunk.strip())  # Save the last chunk

    # Ensure no chunk is too small
    final_merged_sections = []
    buffer_chunk = ""

    for section in merged_sections:
        if len(section) < 1000:  # Small threshold for merging tiny chunks
            buffer_chunk += " " + section.strip()
        else:
            if buffer_chunk:
                final_merged_sections.append(buffer_chunk.strip())  # Save buffered chunk
                buffer_chunk = ""
            final_merged_sections.append(section.strip())

    if buffer_chunk:
        final_merged_sections.append(buffer_chunk.strip())  # Save last buffered chunk

    return final_merged_sections


# Convert Text into Embeddings & Store in FAISS
def store_embeddings(text_chunks, source):
    """Stores text embeddings in FAISS with Redis caching & avoids duplicates."""
    global last_faiss_id  
    model = get_embedding_model()
    new_vectors = []
    new_ids = []

    # Set priority: Excel = 1, PDF = 0
    priority = 1 if source == "Excel" else 0  

    for chunk in text_chunks:        
        # If text exists in FAISS, skip re-adding
        existing_faiss_id = next((key for key, val in faiss_text_store.items() if val["text"] == chunk), None)
        if existing_faiss_id:
            continue  # Skip duplicate embeddings

        embedding = model.encode(chunk).tolist()      
        
        new_id = last_faiss_id
        last_faiss_id += 1  

        # Store with priority flag
        faiss_text_store[str(new_id)] = {"text": chunk, "source": source, "priority": priority}
        new_vectors.append(np.array(embedding).astype('float32'))
        new_ids.append(new_id)

    if new_vectors:
        index.add_with_ids(np.array(new_vectors), np.array(new_ids))
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open(TEXT_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(faiss_text_store, f, ensure_ascii=False, indent=4)
        print(f"Added {len(new_vectors)} new embeddings and updated FAISS index.")
    else:
        print("⚠️ No new embeddings added. FAISS and text store are up to date.")




#Finds the most relevant text chunks using FAISS and keyword expansion
def retrieve_relevant_text(query, top_k=5):
    query = expand_query_with_synonyms(query)
    print("Expanded query", query)
    model = get_embedding_model()
    query_embedding = model.encode(query).reshape(-1)  # Ensure 1D embedding

    # Expand FAISS search using more results
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k * 2)  

    valid_entries = []
    query_keywords = query.lower().split()  # Extract individual words

    for idx, distance in zip(indices[0], distances[0]):  
        if str(idx) in faiss_text_store:
            entry = faiss_text_store[str(idx)]
            entry_embedding = model.encode(entry["text"]).reshape(-1)  
            similarity = 1 - cosine(query_embedding, entry_embedding)

            # Boost score if any keyword matches
            keyword_match = sum(1 for word in query_keywords if word in entry["text"].lower())
            keyword_boost = 0.05 * keyword_match  

            valid_entries.append({
                "text": entry["text"], 
                "priority": entry["priority"],  
                "similarity": similarity + keyword_boost  # Boost similarity if keyword matches
            })

    # Sort results by priority first, then similarity
    sorted_entries = sorted(valid_entries, key=lambda x: (x["priority"], x["similarity"]), reverse=True)

    return "\n\n".join(entry["text"] for entry in sorted_entries) if sorted_entries else "No relevant information found."

# Replaces words in the query with their synonyms to improve search.
def expand_query_with_synonyms(query):
    words = query.lower().split()  # Convert query to lowercase and split into words
    expanded_words = []
    for word in words:
        if word in SYNONYM_MAP:
            expanded_words.extend(SYNONYM_MAP[word])  # Add synonyms
        else:
            expanded_words.append(word)
    return " ".join(expanded_words)  # Return the expanded query as a string

# Function to store question and response in Redis with embeddings
def store_in_cache(question, response):
    embedding = get_embedding_model().encode(question)  # Convert question to vector
    # Convert NumPy array to list before storing in Redis
    redis_client.set(f"question:{question}", json.dumps({
        "embedding": embedding.tolist(),  # Convert NumPy array to list
        "response": response,
        "timestamp": int(time.time())  # Ensure timestamp is stored
    }))

# Function to retrieve similar cached response
def get_cached_response(question, similarity_threshold=0.85):
    query_embedding = get_embedding_model().encode(question)  # Convert new query to vector    
    best_match = None
    highest_similarity = 0

    # Iterate through stored questions in Redis
    for key in redis_client.scan_iter("question:*"):
        data = json.loads(redis_client.get(key))
        stored_embedding = np.array(data["embedding"])
        similarity = 1 - cosine(query_embedding, stored_embedding)

        if similarity > highest_similarity and similarity >= similarity_threshold:
            highest_similarity = similarity
            best_match = data["response"]

    return best_match



# Step 6: Query GPT with Retrieved Context
def query_pdf_assistant(user_query):
    relevant_text = retrieve_relevant_text(user_query, top_k=5)
    chat_history = format_chat_history()
    
    prompt = f"""Document: 'Relevant Section from Brochure'\n\nRelevant Sections:\n{relevant_text} {f"Chat History:\n{chat_history}" if chat_history else ""} \n\nQuestion: {user_query}"""
    
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": (
                "You are an AI assistant answering queries based on Kaplan International Prospectus. Only use the information retrieved from the Brochure. "
                "If information not found, inform to contect Kaplan website or Kaplan staff at its locations."
                "Respond in the same language as the question unless instructed otherwise."
				"Format all numbers and fee components correctly. Use proper spacing in currency values. "
                "For tabular data, return it in a readable format using Markdown-style tables or line breaks. "
                "Do not italicize numbers. Use bold for important details."
                "Format the response in text."
                "Provide the fee or cost breakdown if any."
                "Provide contact details of Kaplan when required."
            )},
            {"role": "user", "content": prompt}
        ]
    ) 
    return response.choices[0].message.content
    #return st.text(relevant_text)
    #return prompt

# Format chat history for OpenAI API to maintain context."
def format_chat_history():
    chat_history = ""
    for message in st.session_state.messages[-5:]:  # Send last 5 messages for context
        role = "User" if message["role"] == "user" else "Assistant"
        chat_history += f"{role}: {message['content']}\n"
    return chat_history


def get_response(user_query):
    #Check Redis cache for existing response.
    cached_response = get_cached_response(user_query)
    
    if cached_response:
        print("Response from cache ", cached_response)
        return cached_response  # Return cached response if found

    # Fetch response from GPT
    response = query_pdf_assistant(user_query)
    print("Response from GPT: ", type(response))

    # Convert response to a string (ensures it's Redis-compatible)
    if not isinstance(response, str):  
        response = str(response)

    # Store new question in cache
    store_in_cache(user_query, response)
    return response

# Extarct data and embeddings
if not index.ntotal:  # Only process if FAISS is empty
    redis_client.flushdb() # clean the Redis DB for every update on the app.
    print("Step 3:", time.strftime('%Y-%m-%d %H:%M:%S'))
    pdf_text = extract_text_from_pdf("Kaplan-International-Prospectus-2024.pdf")
    print("Step 4:", time.strftime('%Y-%m-%d %H:%M:%S'))
    df = pd.read_excel("UniversityPrograms.xlsx")
    UNIVERSITIES = df["university"].dropna().unique().tolist()
    redis_client.set("UNIVERSITIES", json.dumps(UNIVERSITIES))   
    print("Step 5:", time.strftime('%Y-%m-%d %H:%M:%S'))
    text_chunks = split_text_by_headings(pdf_text, 4000, 200)
	store_embeddings(text_chunks,"PDF")
    print("Step 6:", time.strftime('%Y-%m-%d %H:%M:%S'))


# Streamlit UI for Uploading PDF and Asking Questions
st.title("📖 Kaplan AI Course Assistant")
st.write("I am happy to help you on the courses offerred at Kaplan for international students.")


if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores chat messages
else:
    # Display chat messages from session state in order (oldest at top)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.text(message["content"])

# Get user input
if user_input := st.chat_input("Enter your question..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):  # Append Question to session
        st.text(user_input)

    # Generate chatbot response
    response = get_response(user_input)
    response = response.replace("*","")   # Post processing to remove * styling from response
    st.session_state.messages.append({"role": "assistant", "content": response}) # Append responses to session

    # Display chatbot's response
    with st.chat_message("assistant"):
        st.text(response)