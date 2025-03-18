import asyncio
import csv
import os
import hashlib
import re
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Helper function to preprocess text
def preprocess_text(text):
    """
    Clean and normalize the text by removing extra whitespace.
    
    Args:
        text (str): The raw text to preprocess.
    Returns:
        str: The cleaned and normalized text.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = text.strip()  # Remove leading and trailing whitespace
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size):
    """
    Split the text into chunks based on sentence boundaries, aiming for approximately chunk_size words per chunk.
    
    Args:
        text (str): The text to split.
        chunk_size (int): The target number of words per chunk.
    Returns:
        list: A list of text chunks.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        sent_word_count = len(words)
        if current_word_count + sent_word_count > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sent_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sent_word_count
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Function to load and preprocess documents
def load_and_preprocess_documents(directory="./markdown_outputs", chunk_size=200):
    """
    Load markdown files from the specified directory, preprocess the text, and split into chunks.
    
    Args:
        directory (str): The directory containing markdown files (default: "./markdown_outputs").
        chunk_size (int): The target number of words per chunk (default: 200).
    Returns:
        list: A list of text chunks from all documents.
    """
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                clean_text = preprocess_text(content)
                chunks = split_text_into_chunks(clean_text, chunk_size)
                documents.extend(chunks)
    return documents

# Function to build the index
def build_index(documents, index_file="index.faiss"):
    """
    Build an embedding-based search index from the provided documents and save it to a file.
    
    Args:
        documents (list): A list of text chunks to index.
        index_file (str): The file path to save the index (default: "index.faiss").
    Returns:
        tuple: (faiss.IndexFlatIP, list, SentenceTransformer) - The index, documents, and embedding model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)
    embeddings = np.array(embeddings).astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    print(f"Index saved to {index_file}")
    return index, documents, model

def retrieve_relevant_documents(index, documents, model, query, k=5):
    """
    Retrieve the top-k most relevant documents based on the query.
    
    Args:
        index (faiss.IndexFlatIP): The FAISS index containing document embeddings.
        documents (list): The list of document text chunks.
        model (SentenceTransformer): The embedding model ('all-MiniLM-L6-v2').
        query (str): The query string.
        k (int): The number of documents to retrieve (default: 5).
    Returns:
        list: A list of (document, similarity) tuples, sorted by similarity.
    """
    query_embedding = model.encode([query])[0].astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1)
    similarities, indices = index.search(query_embedding, k)
    results = [(documents[idx], similarities[0][i]) for i, idx in enumerate(indices[0])]
    return results

async def main():
    index_flag_file = "index_built.flag"
    documents = []
    
    if os.path.exists(index_flag_file):
        print("Index already built. Loading index for retrieval.")
        index = faiss.read_index("index.faiss")
        documents = load_and_preprocess_documents()
        model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        with open('links.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            links = [row[0] for row in reader][:10]
        browser_config = BrowserConfig(verbose=True)
        run_config = CrawlerRunConfig(
            word_count_threshold=10,
            excluded_tags=['form', 'header'],
            exclude_external_links=True,
            process_iframes=True,
            remove_overlay_elements=True,
            cache_mode=CacheMode.ENABLED
        )
        successful_saves = 0
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for url in links:
                result = await crawler.arun(url=url, config=run_config)
                if result.success:
                    save_markdown(url, result.markdown)
                    successful_saves += 1
                else:
                    print(f"Crawl failed for {url}: {result.error_message}")
        print(f"Saved markdown for {successful_saves} out of {len(links)} URLs.")
        documents = load_and_preprocess_documents()
        index, documents, model = build_index(documents)
        print("Index built and saved successfully.")
        with open(index_flag_file, 'w') as f:
            f.write("Index built successfully.")

    # Initialize the Hugging Face Inference Client
    client = InferenceClient(
        provider="hf-inference",
        api_key="hf"  # Replace with your actual Hugging Face API key
    )

    # Interactive query loop
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        top_docs = retrieve_relevant_documents(index, documents, model, query, k=5)
        print("\nTop 5 relevant documents:")
        for i, (doc, score) in enumerate(top_docs, 1):
            print(f"{i}. Similarity: {score:.4f}\n   Text: {doc[:100]}...\n")
        
        context = "\n\n".join([doc for doc, score in top_docs])
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Please provide a detailed and accurate answer based only on the information in the context above:"""
        
        # Generate answer using the Inference API
        try:
            result = client.text_generation(
                prompt=prompt,
                model="google/gemma-2-2b-it",
                max_new_tokens=250,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )
            answer = result
        except Exception as e:
            print(f"An error occurred while generating the answer: {e}")
            answer = "Sorry, I couldn't generate an answer at this time."
        
        print("\nGenerated Answer:", answer)

def save_markdown(url, markdown_content, output_dir="./markdown_outputs"):
    contact_us_pattern = re.compile(r"(#### Contact Us.*)", re.DOTALL)
    markdown_content = re.sub(contact_us_pattern, '', markdown_content)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = hashlib.md5(url.encode("utf-8")).hexdigest() + ".md"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)

if __name__ == "__main__":
    asyncio.run(main())