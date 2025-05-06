import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import time
import sys 
from sentence_transformers import SentenceTransformer, util
import config 

from rag_core import (
    initialize_agent,          
    retrieve_relevant_documents, 
    generate_answer           
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logging.getLogger("sentence_transformers").setLevel(logging.WARNING) # Quieter logs

# --- Configuration ---
QA_DATASET_FILE = "generated_qa_dataset_ranked.xlsx" # Input dataset for this script
OUTPUT_RESULTS_FILE = "generation_evaluation_results.csv" 

# --- Main Evaluation Logic --- #
def main():
    logging.info("--- Starting Generation Evaluation ---")
    start_time = time.time()

    # 1. Initialize RAG Agent components using the core function
    index, documents_data, model, client = initialize_agent(
        docs_dir=config.DOCS_DIRECTORY,
        index_file=config.INDEX_FILE,
        embedding_model_name=config.EMBEDDING_MODEL,
        chunk_size=config.CHUNK_SIZE
    )

    # 2. Load Evaluation Dataset
    if not os.path.exists(QA_DATASET_FILE):
        logging.error(f"Evaluation dataset not found: {QA_DATASET_FILE}")
        sys.exit(1)
    logging.info(f"Loading evaluation dataset from: {QA_DATASET_FILE}")
    try:
        qa_df = pd.read_excel(QA_DATASET_FILE, engine='openpyxl')
        logging.info(f"Loaded {len(qa_df)} evaluation Q&A pairs.")

    except Exception as e:
        logging.error(f"Failed to load or process dataset {QA_DATASET_FILE}: {e}")
        sys.exit(1)

    # 3. Run Retrieval and Generation for each Question
    evaluation_data = [] # This will store dicts for the results DataFrame
    logging.info(f"Running retrieval and generation for {len(qa_df)} questions...")

    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Evaluating Generation"):
        question = row["Question"]
        ground_truth = row["Answer"] 

        # a) Retrieve context using rag_core function
        retrieved_docs_with_scores = retrieve_relevant_documents(
            index=index,
            documents_data=documents_data,
            model=model,
            query=str(question),
            k=config.RETRIEVAL_K
        )
        # Store context as a list of strings
        retrieved_context_list = [doc for doc, score in retrieved_docs_with_scores]

        # Generate answer using rag_core function
        generated_answer = generate_answer(
            question=str(question),
            context_docs=retrieved_context_list,
            client=client,
            model_name=config.QA_GENERATION_MODEL)
             

        # Store results needed for evaluation and saving
        evaluation_data.append({
            "question": str(question),
            "answer": generated_answer,         
            "contexts": retrieved_context_list,   
            "ground_truth": str(ground_truth)   
        })

    logging.info(f"Generated answers for {len(evaluation_data)} questions.")

    # 4. Create DataFrame from results
    results_df = pd.DataFrame(evaluation_data)
    logging.info("Created DataFrame with generated results.")

    # 5. Calculate Semantic Similarity
    logging.info("Calculating Semantic Similarity between generated and ground truth answers...")
    # Use the same embedding model initialized earlier ('model')
    generated_answers = results_df['answer'].tolist() # List of generated answers
    ground_truth_answers = results_df['ground_truth'].tolist() # List of ground truth answers

    try:
        embeddings_generated = model.encode(generated_answers, convert_to_tensor=True, show_progress_bar=True)
        embeddings_ground_truth = model.encode(ground_truth_answers, convert_to_tensor=True, show_progress_bar=True)

        cosine_scores = util.cos_sim(embeddings_generated, embeddings_ground_truth) 
        semantic_similarity_scores = [cosine_scores[i, i].item() for i in range(len(generated_answers))] # List of semantic similarity scores

        results_df['semantic_similarity'] = semantic_similarity_scores
        mean_similarity = np.mean(semantic_similarity_scores)
        logging.info(f"Mean Semantic Similarity: {mean_similarity:.4f}")

    except Exception as e:
        logging.error(f"Failed to calculate semantic similarity: {e}")
        results_df['semantic_similarity'] = np.nan


    # 6. Report Final Results
    logging.info("--- Overall Generation Evaluation Metrics ---")
    print(f"Evaluated on {len(results_df)} questions.")

    # Calculate and print means
    mean_similarity = results_df['semantic_similarity'].mean()

    print(f"Mean Semantic Similarity: {mean_similarity:.4f}")

    end_time = time.time()
    logging.info(f"Evaluation finished in {end_time - start_time:.2f} seconds.")

    # 7. Save Detailed Results
    try:
        # Convert contexts list to string for easier CSV storage
        results_df['contexts'] = results_df['contexts'].apply(lambda x: "\\n---\\n".join(map(str, x)))

        cols_to_save = ['question', 'semantic_similarity','answer', 'ground_truth','contexts']

        results_df[cols_to_save].to_csv(OUTPUT_RESULTS_FILE, index=False)
        logging.info(f"Detailed evaluation results saved to {OUTPUT_RESULTS_FILE}")
    except Exception as e:
        logging.error(f"Failed to save detailed results to {OUTPUT_RESULTS_FILE}: {e}")


if __name__ == "__main__":
    main() 