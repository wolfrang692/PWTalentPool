# import streamlit as st
# import numpy as np
# import faiss
# import openai
# from tqdm import tqdm
# import json
# import os
#
# # Configuration and file paths - Modify these paths according to where your Streamlit app will access them
# json_file_path = './resumes.json'  # Adjusted path for Streamlit app
# embeddings_file = './embeddings.npy'  # Adjusted path for Streamlit app
# metadata_file = './metadata.json'  # Adjusted path for Streamlit app
# openai.api_key = st.secrets["openai_api_key"]
#

import streamlit as st
import numpy as np
import faiss
import openai
import json
from tqdm import tqdm
import os

# Assuming these are your adjusted paths for Streamlit Cloud deployment
json_file_path = 'resumes.json'
embeddings_file = 'embeddings.npy'
metadata_file = 'metadata.json'
openai.api_key = st.secrets["openai_api_key"]

@st.experimental_singleton
def load_embeddings_and_metadata(embeddings_file, metadata_file):
    if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
        embeddings = np.load(embeddings_file, allow_pickle=True)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)
        return embeddings, metadata, faiss_index
    else:
        raise FileNotFoundError("Embeddings or metadata file not found.")

# Load or generate embeddings and metadata
embeddings, metadata, faiss_index = load_embeddings_and_metadata(embeddings_file, metadata_file)

# Load resumes from JSON file
def load_resumes_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to chunk resumes data for embedding generation
def chunk_resumes_data(resumes_data, chunk_size=100, overlap=10):
    chunks = []
    for resume in resumes_data:
        text = resume['body']
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            chunks.append({
                'document_number': resume['document_number'],
                'name': resume['name'],
                'text': chunk_text
            })
    return chunks

# Function to save embeddings and metadata to files
def save_embeddings_and_metadata(embeddings, metadata, embeddings_file, metadata_file):
    np.save(embeddings_file, embeddings)
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)

@st.cache_data  # Updated to the new caching command
def load_embeddings_and_metadata(embeddings_file, metadata_file):
    embeddings = np.load(embeddings_file, allow_pickle=True)
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    return embeddings, metadata, faiss_index


# # Function to generate embeddings for chunks of resumes
# def generate_embeddings(resumes_chunks):
#     embeddings = []
#     for resume_chunk in tqdm(resumes_chunks, desc="Generating Embeddings", disable=True):  # disable tqdm in Streamlit
#         response = openai.Embedding.create(
#             input=resume_chunk['text'],
#             engine="text-embedding-ada-002"
#         )
#         embeddings.append(response['data'][0]['embedding'])
#     return embeddings

def find_similar_chunks(query, faiss_index, embeddings, metadata, n=5):
    response = openai.Embedding.create(input=query, engine="text-embedding-ada-002")
    query_embedding = response['data'][0]['embedding']
    _, I = faiss_index.search(np.array([query_embedding]).astype('float32'), n)

    try:
        # Adjusting to correctly process metadata as a list of lists
        return [(metadata[i][0], metadata[i][1]) for i in I[0]]
    except Exception as e:
        st.error(f"Error processing metadata with indices: {e}")
        return []

def extract_text_around_keywords(text, keywords, window=50):
    """
    Extracts parts of text around specified keywords.
    - text: The full text from which to extract snippets.
    - keywords: A list of keywords to search for in the text.
    - window: Number of words to include before and after the keyword.
    """
    import re
    words = text.split()
    extracted_texts = []
    for keyword in keywords:
        for match in re.finditer(r'\b{}\b'.format(re.escape(keyword)), text, flags=re.IGNORECASE):
            start_pos = max(match.start() - window, 0)
            end_pos = min(match.end() + window, len(text))
            start_word = text.rfind(' ', 0, start_pos) + 1  # start from the beginning of the word
            end_word = text.find(' ', end_pos)  # end at the last whole word
            if end_word == -1: end_word = len(text)
            snippet = text[start_word:end_word]
            extracted_texts.append(snippet)
    return ' ... '.join(extracted_texts)

def generate_response_with_gpt(query, similar_metadata, resumes_data):
    if similar_metadata:
        messages = [{
            "role": "system",
            "content": "You are an HR expert analyzing candidates' resumes and answering queries based on their profiles."
        }]
        query_keywords = query.split()  # simplistic split, consider more sophisticated NLP techniques for better results

        for document_number, candidate_name in similar_metadata:
            candidate_profile_summary = next(
                (resume['body'] for resume in resumes_data if resume['document_number'] == document_number), None)
            if candidate_profile_summary:
                extracted_text = extract_text_around_keywords(candidate_profile_summary, query_keywords)
                messages[0][
                    'content'] += f"\n\nFor {candidate_name}, here is a relevant section of their profile: '{extracted_text}'."

        messages.append({
            "role": "user",
            "content": query
        })
    else:
        messages = [{
            "role": "system",
            "content": "You are an HR expert analyzing candidates' resumes and answering queries based on their profiles."
        }, {
            "role": "user",
            "content": f"I couldn't find anyone with '{query}' in their resume. Could you specify other skills or details?"
        }]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message['content']



# Streamlit application start
st.title('PW Talent Pool Query Engine v1.1')

# Load resumes data
resumes_data = load_resumes_from_json(json_file_path)

# # Load or generate embeddings and metadata
# if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
#     embeddings, metadata, faiss_index = load_embeddings_and_metadata(embeddings_file, metadata_file)
# else:
#     resumes_chunks = chunk_resumes_data(resumes_data)
#     embeddings = generate_embeddings(resumes_chunks)
#     metadata = [{'document_number': chunk['document_number'], 'name': chunk['name']} for chunk in resumes_chunks]
#     save_embeddings_and_metadata(np.array(embeddings).astype('float32'), metadata, embeddings_file, metadata_file)
#     embeddings, metadata, faiss_index = load_embeddings_and_metadata(embeddings_file, metadata_file)

# Streamlit UI for query input
user_query = st.text_input("Enter your query about candidate skills, experience, etc:")

if user_query:
    with st.spinner('Searching for candidates and generating response...'):
        similar_metadata = find_similar_chunks(user_query, faiss_index, embeddings, metadata)
        # Adjust the structure of similar_metadata if necessary to match your function's expectation
        similar_metadata_structured = [(doc_num, name) for doc_num, name in similar_metadata]
        chat_response = generate_response_with_gpt(user_query, similar_metadata_structured, resumes_data)
        st.text_area("Chatbot Response:", value=chat_response, height=300)
#
