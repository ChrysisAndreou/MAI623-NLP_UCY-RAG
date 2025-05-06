import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from together import Together
import logging

# Import utility functions
from utils import load_documents 

# Function to build the index
def build_index(documents_data, model, index_file="index.faiss"):
    """
    Build an embedding-based search index from the provided document data and save it.

    Args:
        documents_data (list[dict]): A list of document dictionaries [{'text': str, 'source_file': str}, ...].
        model (SentenceTransformer): The embedding model instance.
        index_file (str): The file path to save the index (default: "index.faiss").

    Returns:
        faiss.IndexFlatIP | None: The created FAISS index, or None if building fails.
    """
    if not documents_data:
        logging.warning("Cannot build index: No documents provided.")
        return None

    document_texts = [doc['text'] for doc in documents_data] # chunked text
    logging.info(f"Encoding {len(document_texts)} document chunks for indexing...")

    try:
        embeddings = model.encode(document_texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Normalize embeddings: This scales vectors to unit length, making the Inner Product
        # calculated by IndexFlatIP equivalent to Cosine Similarity.
        # Normalize embeddings for Inner Product (IP) index (cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1e-10, norms) # replace zero norms with 1e-10 to avoid division by zero
        normalized_embeddings = embeddings / safe_norms 

        dimension = normalized_embeddings.shape[1] # shape[0] is the number of chunks, shape[1] is the dimension of the embeddings
        index = faiss.IndexFlatIP(dimension) 
        index.add(normalized_embeddings)

        logging.info(f"Index built with {index.ntotal} vectors.")
        logging.info(f"Saving index to {index_file}...")
        faiss.write_index(index, index_file)
        logging.info("Index saved successfully.")
        return index

    except Exception as e:
        logging.error(f"Error building or saving index: {e}")
        return None


def retrieve_relevant_documents(index, documents_data, model, query, k=5):
    """
    Retrieve the top-k most relevant document texts based on the query.

    Args:
        index (faiss.IndexFlatIP): The FAISS index containing document embeddings.
        documents_data (list[dict]): The list of document dictionaries [{'text': str, 'source_file': str}, ...].
        model (SentenceTransformer): The embedding model instance.
        query (str): The query string.
        k (int): The number of documents to retrieve (default: 5).

    Returns:
        list[tuple[str, float]]: A list of (document_text, similarity_score) tuples, sorted by similarity.
                                 Returns empty list if index or documents are empty/invalid.
    """
    if not documents_data or index is None or index.ntotal == 0:
        logging.warning("Retrieval skipped: Invalid index or empty document list.")
        return []
    
    document_texts = [doc['text'] for doc in documents_data]

    try:
        # Encode the query 
        query_embedding = model.encode([query], show_progress_bar=False)[0].astype('float32')
        
        # Normalize the query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0: query_norm = 1e-10 # Avoid division by zero
        normalized_query_embedding = query_embedding / query_norm
        normalized_query_embedding = normalized_query_embedding.reshape(1, -1) # reshape to 1D array for FAISS

        # Search the index
        similarities, indices = index.search(normalized_query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]): # indices[0] is the list of indices of the top-k most similar chunks
            results.append((document_texts[idx], similarities[0][i])) # similarities[0][i] is the similarity score of the i-th most similar chunk

        # Sort by similarity (descending) - FAISS IP index should already return in order, but doesn't hurt to ensure
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    except Exception as e:
        logging.error(f"Error during document retrieval: {e}")
        return []


def initialize_agent(docs_dir="./long_files", index_file="index.faiss", embedding_model_name='all-MiniLM-L6-v2', chunk_size=200):
    """
    Initializes the RAG agent components.

    Args:
        docs_dir (str): Directory containing source markdown documents.
        index_file (str): Path to the FAISS index file.
        embedding_model_name (str): Name of the SentenceTransformer model.
        chunk_size (int): Chunk size for document processing.

    Returns:
        tuple: (faiss.IndexFlatIP | None, list[dict], SentenceTransformer | None, Together | None)
               - The loaded/built index (or None on error).
               - The list of document data dictionaries.
               - The initialized embedding model (or None on error).
               - The initialized Together client (or None on error).
               Returns (None, [], None, None) if critical initialization fails (model load).
    """
    logging.info("--- Initializing RAG Agent Core ---")

    # 1. Initialize Embedding Model
    model = None
    try:
        logging.info(f"Loading embedding model: {embedding_model_name}")
        model = SentenceTransformer(embedding_model_name)
        logging.info("Embedding model loaded.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to load embedding model '{embedding_model_name}': {e}")
        return None, [], None, None # Critical failure

    # 2. Initialize Together Client
    client = None
    try:
        client = Together()
        logging.info("Together client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Together client: {e}")
        pass # Keep client as None

    # 3. Load Documents 
    logging.info(f"Loading documents from {docs_dir} with chunk size {chunk_size}...")
    documents_data = load_documents(directory=docs_dir, chunk_size=chunk_size) # Returns list[dict]
    
    # 4. Load or Build Index
    index = None
    model_dim = model.get_sentence_embedding_dimension()
    if os.path.exists(index_file):
        logging.info(f"Attempting to load existing index from {index_file}...")
        try:
            index = faiss.read_index(index_file)
        except Exception as e:
            logging.error(f"Error loading index file {index_file}: {e}. Will attempt to build.")
            index = None
    else:
        logging.info(f"Index file {index_file} not found. Will attempt to build.")

    # Attempt to build index if not loaded 
    if index is None:
        logging.info(f"Building index for {len(documents_data)} document chunks...")
        # Pass the initialized model and the list of dicts to build_index
        index = build_index(documents_data, model, index_file=index_file)
        if index:
            logging.info("Index built successfully.")
        else:
            logging.error("Failed to build index.")
            
    # --- Add logging here ---
    if index:
        logging.info(f"Index is ready. Type: {type(index)}, Total vectors: {index.ntotal}")
    else:
        logging.error("Index object is None after loading/building attempt.")
    # --- End added logging ---

    logging.info("--- RAG Agent Core Initialization Complete ---")
    # Return index (might be None), the loaded documents_data (list[dict]), model, and client (might be None)
    return index, documents_data, model, client


def generate_answer(question: str, context_docs: list[str], client: Together, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", max_tokens: int = 350, temperature: float = 0.6, top_p: float = 0.9):
    """
    Generates an answer based on the provided context documents using Together AI.

    Args:
        question (str): The user's question.
        context_docs (list[str]): A list of relevant context strings.
        client (Together): An initialized Together client instance.
        model_name (str): The name of the LLM model to use.
        max_tokens (int): Maximum tokens for the generated answer.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.

    Returns:
        str: The generated answer, or an error message.
    """
    if not client:
        logging.error("generate_answer called without a valid Together client.")
        return "Sorry, the generation service client is not available."
    if not context_docs:
        logging.warning("generate_answer called with no context documents.")
        return "I couldn't find any relevant information in my knowledge base to answer your question."

    # Format context for the prompt
    context = "\n\n".join(context_docs)

    # Construct the prompt messages
    # This prompt aims for precise extraction based only on the context.
    messages = [
        {"role": "system", "content": "You are an extraction engine. Your task is to find the precise answer to the user's question within the provided Context. Respond *only* with the extracted answer, nothing else. Be extremely brief. Do not add any introductory phrases like 'The answer is' or 'Based on the context'."},
        {"role": "user", "content": f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""}
    ]

    answer = None
    try:
        logging.info(f"Generating answer with model '{model_name}' via Together AI...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if response and response.choices:
            generated_message = response.choices[0].message  
            if generated_message and generated_message.content:
                answer = generated_message.content.strip()
                logging.info(f"Successfully generated answer.")
            else:
                logging.error(f"API response from '{model_name}' did not contain valid content: {response}")
                answer = f"Sorry, the model '{model_name}' response format was unexpected (no content)."
        else:
            logging.error(f"API call to '{model_name}' returned an unexpected response structure: {response}")
            answer = f"Sorry, received an unexpected response structure from the model '{model_name}'."

    except Exception as e:
        # Catch potential API errors (e.g., key issues, model not found, network errors)
        logging.error(f"API call to model '{model_name}' failed: {e}")
        answer = f"Sorry, I encountered an error trying to generate an answer using model '{model_name}'. Please check API key and model availability."

    # Fallback if answer is still None
    if answer is None:
        answer = "Sorry, I couldn't generate an answer due to an unexpected issue."

    return answer 