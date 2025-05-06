import os
import re
import logging
from tqdm import tqdm
from llama_index.core.node_parser import SentenceSplitter


def preprocess_text(text):
    """Clean and normalize the text by removing extra whitespace."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_text_into_chunks(text, chunk_size, chunk_overlap=20):
    """
    Split text into chunks using SentenceSplitter from llama-index.
    
    Args:
        text (str): The text to split.
        chunk_size (int): The target number of characters per chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        
    Returns:
        list[str]: A list of text chunks.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_text(text)
    return chunks


def load_documents(directory, chunk_size):
    """
    Load markdown files from a directory, preprocess, chunk, and track source filename.
    
    Args:
        directory (str): The directory containing markdown (.md) files.
        chunk_size (int): The target number of characters per chunk for splitting.
        
    Returns:
        list[dict]: A list of dictionaries with 'text' and 'source_file' keys.
    """
    all_chunks_with_source = []
    
    try:
        filenames = [f for f in os.listdir(directory) 
                    if f.endswith('.md') and os.path.isfile(os.path.join(directory, f))]
        
        logging.info(f"Found {len(filenames)} markdown files in '{directory}'.")
        file_iterator = tqdm(filenames, desc="Loading and Chunking Documents")

        for filename in file_iterator:
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                clean_text = preprocess_text(content)
                text_chunks = split_text_into_chunks(clean_text, chunk_size)

                for chunk_text in text_chunks:
                    all_chunks_with_source.append({
                        "text": chunk_text,
                        "source_file": filename
                    })

            except Exception as e:
                logging.error(f"Error processing file {filepath}: {e}")

        logging.info(f"Generated {len(all_chunks_with_source)} chunks from {len(filenames)} files.")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during document loading: {e}")
        return []

    return all_chunks_with_source