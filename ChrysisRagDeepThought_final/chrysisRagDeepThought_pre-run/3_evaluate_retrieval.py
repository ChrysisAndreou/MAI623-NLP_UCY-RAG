import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import faiss
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from tqdm import tqdm
import logging
import time
import sys # For exit
import json 
import config 

from rag_core import retrieve_relevant_documents, initialize_agent 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# --- Configuration --- 
QA_DATASET_FILE = "generated_qa_dataset_ranked.xlsx" 
EVAL_K_VALUES = [1, 3, 5] 
MAX_K = max(EVAL_K_VALUES) # retrieve only the top k documents

# --- Evaluation Functions --- 

def calculate_hit_rate(retrieved_chunks, source_chunk, k):
    """Check if the source_chunk is within the top-k retrieved chunks."""
    return 1 if source_chunk in retrieved_chunks[:k] else 0

def calculate_reciprocal_rank(retrieved_chunks, source_chunk, k):
    """Calculate the reciprocal rank of the source_chunk within top-k."""
    try:
        rank = retrieved_chunks[:k].index(source_chunk) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0

def calculate_precision(retrieved_chunks, ground_truth_relevant_set, k):
    """Calculate precision@k."""
    if k == 0: return 0.0
    top_k_retrieved = set(retrieved_chunks[:k]) # Convert to set for faster intersection
    relevant_found = len(top_k_retrieved.intersection(ground_truth_relevant_set))
    return relevant_found / k

def calculate_kendall_tau(retrieved_chunks, ground_truth_ranked_list, k):
    """
    Calculate Kendall's Tau-b @ k.
    Compares the ranking of shared items between the retrieved list (top k)
    and the ground truth list.
    """
    top_k_retrieved = retrieved_chunks[:k]
    ground_truth_set = set(ground_truth_ranked_list)

    # Identify items present in both lists, maintaining retrieved order
    shared_items = [item for item in top_k_retrieved if item in ground_truth_set]

    if len(shared_items) < 2:
        # Kendall's Tau is undefined or 0 for less than 2 shared items
        return 0.0

    # Create rank arrays ONLY for shared items
    # Rank in retrieved list (based on position in top_k_retrieved)
    retrieved_ranks = [top_k_retrieved.index(item) for item in shared_items]
    # Rank in ground truth list (based on position in ground_truth_ranked_list)
    ground_truth_ranks = [ground_truth_ranked_list.index(item) for item in shared_items]

    try:
        tau, p_value = kendalltau(retrieved_ranks, ground_truth_ranks)
        # Handle NaN result which can occur (e.g., if all ranks are identical)
        return tau if not np.isnan(tau) else 0.0 # Return 0 if tau is NaN
    except Exception as e:
         logging.warning(f"Could not calculate Kendall's Tau: {e}. Returning 0.")
         return 0.0


# --- Main Evaluation Logic --- #
def main():
    logging.info("--- Starting Retrieval Evaluation ---")
    start_time = time.time()

    # 1. Load Q&A Dataset (Input for evaluation)
    if not os.path.exists(QA_DATASET_FILE):
        logging.error(f"Evaluation dataset not found: {QA_DATASET_FILE}")
        sys.exit(1)
    logging.info(f"Loading Q&A dataset from: {QA_DATASET_FILE}")
    try:
        qa_df = pd.read_excel(QA_DATASET_FILE, engine='openpyxl')
        logging.info(f"Loaded {len(qa_df)} evaluation Q&A pairs.")
        # Identify ground truth columns dynamically
        ground_truth_cols = sorted([col for col in qa_df.columns if col.startswith("Relevant_Chunk_Rank_")])
        logging.info(f"Using ground truth columns: {ground_truth_cols}")

    except Exception as e:
        logging.error(f"Failed to load dataset {QA_DATASET_FILE}: {e}")
        sys.exit(1)

    # 2. Initialize Agent Components using rag_core.initialize_agent
    logging.info(f"Initializing agent components (model, docs, index) for evaluation...")
    index, documents_data, model, client = initialize_agent(
        docs_dir=config.DOCS_DIRECTORY,
        index_file=config.INDEX_FILE,
        embedding_model_name=config.EMBEDDING_MODEL,
        chunk_size=config.CHUNK_SIZE
    )


    # 6. Evaluation Loop 
    logging.info(f"Starting evaluation for K values: {EVAL_K_VALUES}")
    results = [] # To store results per question

    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Evaluating Queries"):
        question = row["Question"]
        # Rank 1 is the primary target for Hit Rate, Mean Reciprocal Rank
        source_chunk = row["Relevant_Chunk_Rank_1"]
        # Ground truth includes all ranked relevant chunks for Precision, Kendall Tau
        ground_truth_ranked_list = [row[col] for col in ground_truth_cols if pd.notna(row[col]) and isinstance(row[col], str) and row[col].strip()]
        ground_truth_relevant_set = set(ground_truth_ranked_list) # Convert to set for faster intersection


        retrieved_docs_with_scores = retrieve_relevant_documents(
            index=index,              
            documents_data=documents_data, 
            model=model,                
            query=str(question),        
            k=MAX_K
        )
        # Extract just the text for evaluation metrics
        retrieved_chunks_text = [chunk for chunk, score in retrieved_docs_with_scores]

        # Calculate metrics for this question for each K
        question_results = {'question': question}
        for k in EVAL_K_VALUES:
            hit = calculate_hit_rate(retrieved_chunks_text, source_chunk, k)
            rr = calculate_reciprocal_rank(retrieved_chunks_text, source_chunk, k)
            precision = calculate_precision(retrieved_chunks_text, ground_truth_relevant_set, k)
            kendall = calculate_kendall_tau(retrieved_chunks_text, ground_truth_ranked_list, k)

            question_results[f'hit_rate@{k}'] = hit
            question_results[f'mrr@{k}'] = rr
            question_results[f'precision@{k}'] = precision
            question_results[f'kendall_tau@{k}'] = kendall

        results.append(question_results)

    # 7. Aggregate and Report Results
    results_df = pd.DataFrame(results)
    logging.info("\n--- Overall Retrieval Evaluation Metrics ---")
    print(f"Evaluated on {len(results_df)} questions.")

    aggregated_metrics = {}
    for k in EVAL_K_VALUES:
        # Calculate means for each metric at k
        # Use float() to ensure JSON serializability of numpy floats
        metrics_k = {
            f'Mean Hit Rate @{k}': float(results_df[f'hit_rate@{k}'].mean()),
            f'MRR @{k}': float(results_df[f'mrr@{k}'].mean()),
            f'Mean Precision @{k}': float(results_df[f'precision@{k}'].mean()),
            f'Mean Kendall Tau-b @{k}': float(results_df[f'kendall_tau@{k}'].mean()) # Note: Tau-b
        }
        aggregated_metrics[k] = metrics_k
        print(f"\n--- Metrics for K={k} ---")
        for name, value in metrics_k.items():
            print(f"{name}: {value:.4f}")

    # Save aggregated metrics to JSON
    summary_file = "retrieval_evaluation_summary.json"
    try:
        with open(summary_file, 'w') as f:
            json.dump(aggregated_metrics, f, indent=4)
        logging.info(f"Aggregated retrieval metrics saved to {summary_file}")
    except Exception as e:
        logging.error(f"Failed to save retrieval summary: {e}")

    end_time = time.time()
    logging.info(f"Evaluation finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main() 