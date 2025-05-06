import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from together import Together
from tqdm import tqdm
import math
import logging
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Import shared utility functions and core components
from utils import load_documents
from rag_core import initialize_agent, retrieve_relevant_documents
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# --- Configuration Specific to this Script ---
TOTAL_QA_PAIRS = 39 # Target number of Q&A pairs for generation phase
INTERMEDIATE_OUTPUT_FILE = "generated_qa_dataset.xlsx" # Output file from generation phase
FINAL_OUTPUT_FILE = "generated_qa_dataset_ranked.xlsx" # Output file from augmentation phase
CLUSTER_PLOT_DIR = "cluster_plots" # Directory for cluster plots


# --- Helper Function specific to QA Generation --- #

def generate_qa_pair(client, chunk_text, model_name, max_retries=3):
    """Generate a Q&A pair from a chunk using the specified Together AI model with retries."""
    
    for attempt in range(max_retries):
        prompt = f"""Context:
{chunk_text}

Based *only* on the text provided in the Context above, generate:
1. A relevant question that can be answered solely from this text.
2. The answer to that question, extracted directly from the text.

IMPORTANT: Format your response exactly as:
Question: [your question here]
Answer: [your answer here]"""

        messages = [{"role": "user", "content": prompt}]
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.5,
                top_p=0.9,
            )
            generated_text = response.choices[0].message.content.strip()

            question_match = re.search(r"Question:\s*(.*?)\s*Answer:", generated_text, re.IGNORECASE | re.DOTALL)
            answer_match = re.search(r"Answer:\s*(.*)", generated_text, re.IGNORECASE | re.DOTALL)

            question = question_match.group(1).strip() if question_match else None
            answer = answer_match.group(1).strip() if answer_match else None

            if question and answer:
                logging.info(f"Successfully generated Q&A pair on attempt {attempt+1}")
                return question, answer
            else:
                logging.warning(f"Attempt {attempt+1}: Could not parse Q&A from response. Retrying...")
        
        except Exception as e:
            logging.error(f"Attempt {attempt+1}: Error during generation: {e}")
            
    # If we get here, all retries failed
    logging.error(f"Failed to generate parseable Q&A pair after {max_retries} attempts for chunk '{chunk_text[:50]}...'")
    return None, None

# --- Helper function for Visualization ---
def plot_clusters(documents_data_with_tsne, selected_indices, k):
    """Generates and saves a t-SNE plot with centroids and selected points."""
    logging.info("Generating t-SNE plot with centroids and selected points...")

    # Create DataFrame for plotting from the enriched documents_data
    plot_data = []
    selected_indices_set = set(selected_indices)
    for i, item in enumerate(documents_data_with_tsne):
        if 'tsne_1' in item and 'tsne_2' in item and 'cluster' in item:
            plot_data.append({
                'tsne_1': item['tsne_1'],
                'tsne_2': item['tsne_2'],
                'cluster': item['cluster'],
                'selected': i in selected_indices_set
            })
        else:
            logging.warning(f"Skipping item {i} in plot generation due to missing t-SNE/cluster data.")

    if not plot_data:
        logging.error("Cannot generate plot: No valid data points with t-SNE coordinates.")
        return

    df_plot = pd.DataFrame(plot_data)

    # Calculate 2D centroids (mean of t-SNE coordinates per cluster)
    centroids_2d = df_plot.groupby('cluster')[['tsne_1', 'tsne_2']].mean().reset_index()

    # Create plot directory if it doesn't exist
    if not os.path.exists(CLUSTER_PLOT_DIR):
        os.makedirs(CLUSTER_PLOT_DIR)
        logging.info(f"Created plot directory: {CLUSTER_PLOT_DIR}")

    # Plotting
    plt.figure(figsize=(14, 10))

    palette = sns.color_palette("hsv", n_colors=k)
    # Plot all points colored by cluster
    scatter = sns.scatterplot(
        x="tsne_1", y="tsne_2",
        hue="cluster",
        palette=palette, 
        data=df_plot,
        legend="full",
        alpha=0.5, # Make non-selected points semi-transparent
        s=50 # Default size for non-selected points
    )

    # Highlight selected points
    selected_points = df_plot[df_plot['selected']]
    plt.scatter(
        selected_points['tsne_1'], selected_points['tsne_2'],
        marker='o', # Circle marker
        c=selected_points['cluster'].apply(lambda cluster_id: palette[cluster_id] if 0 <= cluster_id < len(palette) else '#000000'), # Use apply to map cluster id to palette color 
        s=150,        # Larger size
        edgecolor='black', # Black edge for visibility
        linewidth=1,
        label='Selected for Q&A' 
    )

    # Plot 2D centroids
    plt.scatter(
        centroids_2d['tsne_1'], centroids_2d['tsne_2'],
        marker='X',      # Cross marker
        c=centroids_2d['cluster'].apply(lambda cluster_id: palette[cluster_id] if 0 <= cluster_id < len(palette) else '#000000'),
        s=200, edgecolor='black', linewidth=1.5,     
        label='Cluster Centroid (2D Mean)'
    )

    # Create a combined legend
    handles, labels = scatter.get_legend_handles_labels()

    plt.title(f't-SNE Visualization of Document Chunk Clusters (k={k})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    # Place legend outside the plot
    plt.legend(handles=handles, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    plot_filename = os.path.join(CLUSTER_PLOT_DIR, f"kmeans_tsne_clusters_k{k}_highlighted.png")
    try:
        plt.savefig(plot_filename)
        logging.info(f"Saved highlighted cluster visualization to {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to save highlighted cluster plot: {e}")
    plt.close()

# --- Dataset Generation Logic --- #

def run_qa_generation(client, model_encoder, documents_data):
    """Loads documents, clusters, samples, generates Q&A pairs, and saves intermediate file."""
    logging.info("--- Starting QA Generation Phase ---")

    # 1. Use pre-loaded documents
    all_chunks_text = [item["text"] for item in documents_data]
    logging.info(f"Using {len(documents_data)} pre-loaded document chunks.")

    # 2. Embed Chunks using the provided model_encoder
    logging.info(f"Generating embeddings for {len(all_chunks_text)} chunks using provided model...")
    embeddings = model_encoder.encode(all_chunks_text, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    logging.info(f"Embeddings generated with shape: {embeddings.shape}")

    # 3. Cluster Chunks
    num_chunks = len(all_chunks_text)
    # k clusters is the square root of the number of chunks
    k = min(max(2, int(np.sqrt(num_chunks))), num_chunks)
    logging.info(f"Performing K-Means clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init is the number of times the k-means algorithm will be run with different centroid seeds
    cluster_labels = kmeans.fit_predict(embeddings) # fit_predict returns the cluster labels for each chunk
    logging.info("Clustering complete.")

    # --- Perform t-SNE Early --- (But plot later)
    tsne_results = None
    logging.info("Performing t-SNE for visualization...")
    perplexity_value = min(30, embeddings.shape[0] - 1)
    if perplexity_value > 0: 
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, max_iter=300)
        tsne_results = tsne.fit_transform(embeddings)
        logging.info("t-SNE transformation complete.")
    else:
        logging.warning(f"Skipping t-SNE calculation: Not enough samples ({embeddings.shape[0]}) for perplexity > 0.")

    # Add cluster labels, embeddings, and t-SNE results to our data structure
    for i, item in enumerate(documents_data):
        item["cluster"] = cluster_labels[i]
        item["embedding"] = embeddings[i] # Store embedding for centroid distance calculation
        if tsne_results is not None:
            item["tsne_1"] = tsne_results[i, 0]
            item["tsne_2"] = tsne_results[i, 1]

    # 4. Proportional Sampling & Centroid Selection
    num_clusters = k
    cluster_counts = pd.Series(cluster_labels).value_counts().to_dict() # returns a dictionary of cluster labels and their counts of chunks in each cluster
    total_chunks_clustered = sum(cluster_counts.values()) # sum of count of chunks in all clusters
    samples_per_cluster = {} # how many chunks should be selected from each cluster for Q&A generation.
    total_samples_planned = 0 # total number of chunks to be selected across all clusters for Q&A generation.

    logging.info("Calculating samples per cluster...")
    for cluster_id in range(num_clusters):
        count = cluster_counts.get(cluster_id, 0) # get the count of chunks in the cluster
        if count > 0:
            proportion = count / total_chunks_clustered # proportion of chunks in the cluster to the total number of chunks in all clusters
            num_samples = round(TOTAL_QA_PAIRS * proportion) # number of chunks to be selected from the cluster
            samples_per_cluster[cluster_id] = max(1, int(num_samples)) # ensure at least 1 chunk is selected from the cluster
            total_samples_planned += samples_per_cluster[cluster_id] # total number of chunks to be selected across all clusters for Q&A generation.
        else:
             samples_per_cluster[cluster_id] = 0

    logging.info(f"Final samples planned per cluster: {samples_per_cluster}")
    logging.info(f"Total samples to select: {total_samples_planned}")
    logging.info("Note: Due to the combination of round() (which can round up) and the max(1, ...) rule (which forces small clusters to contribute at least one sample), the final total_samples_planned (39) can sum up to a number slightly different from the initial TOTAL_QA_PAIRS target (42).")

    logging.warning("""The reason RAG is evaluated on 42 questions is to help answer the ultimate question. 
In the programming literature, 42 is often used as a random number seed in reference to the ultimate question. 
The number 42 is famously known as the 'Answer to the Ultimate Question of Life, The Universe, and Everything' in The Hitchhiker's Guide to the Galaxy by Douglas Adams. 
In the story, 42 is the answer computed by a supercomputer called Deep Thought, but the actual question remains unknown, making the answer cryptic and humorous.""")

    selected_chunks_for_qa = []
    logging.info("Selecting representative chunks near centroids...")
    for cluster_id, num_samples in samples_per_cluster.items():
        if num_samples == 0: continue # skip empty clusters

        # Get indices and embeddings for the current cluster
        cluster_indices = [i for i, item in enumerate(documents_data) if item["cluster"] == cluster_id] # get the indices of the chunks in documents_data that are in the cluster
        if not cluster_indices: continue # skip clusters with no chunks
        cluster_embeddings = embeddings[cluster_indices] # get the embeddings of the chunks in the cluster
        if cluster_embeddings.shape[0] == 0: continue # skip clusters with no embeddings
        if cluster_embeddings.shape[0] < num_samples:
            logging.warning(f"Cluster {cluster_id} has only {cluster_embeddings.shape[0]} chunks, less than the requested {num_samples}. Selecting all.")
            num_samples = cluster_embeddings.shape[0]

        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)

        # Find the chunk closest to the centroid first
        similarities_to_centroid = cosine_similarity(cluster_embeddings, centroid).flatten()
        closest_to_centroid_local_idx = np.argmax(similarities_to_centroid)

        # Initialize list of selected indices within this cluster
        selected_local_indices = [closest_to_centroid_local_idx]
        # add as first chunk the one closest to the centroid 
        # (centroid is not a chunk is just the average of all the chunks embeddings in the cluster)
        selected_embeddings = [cluster_embeddings[closest_to_centroid_local_idx]] 

        # Create a pool of candidate local indices (excluding the first selected one)
        candidate_local_indices = list(range(cluster_embeddings.shape[0]))
        candidate_local_indices.pop(closest_to_centroid_local_idx)

        # Iteratively select the most diverse chunks
        while len(selected_local_indices) < num_samples and candidate_local_indices:
            max_similarities_for_candidates = [] # similarity of each candidate to the most similar already selected chunk

            # Calculate similarity of each candidate to *all already selected* chunks
            candidate_embeddings_matrix = cluster_embeddings[candidate_local_indices] # candidate chunks
            selected_embeddings_matrix = np.array(selected_embeddings) # already selected chunks

            # Shape: (n_candidates, n_selected)
            similarity_matrix = cosine_similarity(candidate_embeddings_matrix, selected_embeddings_matrix)

            # Find the max similarity for each candidate (how similar it is to its closest neighbor among selected)
            if similarity_matrix.shape[1] > 0: # Ensure there are selected embeddings to compare against
                max_similarities_for_candidates = np.max(similarity_matrix, axis=1)
            else: # Should only happen on first iteration if only 1 selected
                 max_similarities_for_candidates = np.zeros(len(candidate_local_indices)) # Treat as 0 similarity if nothing selected yet

            # Find the candidate with the *minimum* maximum similarity (most diverse)
            # among the farthest neighbors (the farthest the better)
            most_diverse_candidate_idx_in_list = np.argmin(max_similarities_for_candidates)
            most_diverse_local_idx = candidate_local_indices[most_diverse_candidate_idx_in_list]

            # Select this chunk
            selected_local_indices.append(most_diverse_local_idx)
            selected_embeddings.append(cluster_embeddings[most_diverse_local_idx])

            # Remove the selected chunk from the candidates
            candidate_local_indices.pop(most_diverse_candidate_idx_in_list)

        # Convert local cluster indices back to original document indices
        original_indices = [cluster_indices[local_idx] for local_idx in selected_local_indices]
        for index in original_indices:
            selected_chunks_for_qa.append(documents_data[index])

    logging.info(f"Selected {len(selected_chunks_for_qa)} chunks for Q&A generation using diversity sampling.")
    # Get the original indices of the final selected chunks
    final_selected_indices = [documents_data.index(chunk) for chunk in selected_chunks_for_qa]

    # --- Perform Visualization Now ---
    if tsne_results is not None:
        plot_clusters(documents_data, final_selected_indices, k)
    else:
        logging.warning("Skipping cluster plot generation as t-SNE was not performed.")
    # --- End Visualization ---

    # 5. Generate Q&A Pairs
    qa_results = []
    logging.info(f"Generating Q&A pairs using model: {config.QA_GENERATION_MODEL}...")
    for chunk_data in tqdm(selected_chunks_for_qa, desc="Generating Q&A"):
        question, answer = generate_qa_pair(client, chunk_data["text"], config.QA_GENERATION_MODEL)
        if question and answer:
            qa_results.append({
                "Question": question,
                "Answer": answer, # This is the ground truth for eval
                "Source_File": chunk_data["source_file"],
                "Source_Chunk": chunk_data["text"]
            })
        else:
            logging.warning(f"Skipping chunk from {chunk_data['source_file']} due to Q&A generation failure.")

    logging.info(f"Successfully generated {len(qa_results)} Q&A pairs.")

    # 6. Store Intermediate Results
    if qa_results:
        logging.info(f"Saving {len(qa_results)} Q&A pairs to intermediate file: {INTERMEDIATE_OUTPUT_FILE}...")
        df = pd.DataFrame(qa_results)
        df.to_excel(INTERMEDIATE_OUTPUT_FILE, index=False, engine='openpyxl')
        logging.info(f"Intermediate dataset saved successfully.")
        return True, documents_data, embeddings
    else:
        logging.warning("No Q&A pairs were generated successfully. Cannot proceed to augmentation.")
        return False, None, None # Signal failure

# --- Dataset Augmentation Logic --- #

def run_qa_augmentation(model_encoder, index, documents_data_with_embeddings, embeddings_array):
    """
    Loads intermediate Q&A, finds neighbors of the Source_Chunk using cosine similarity
    on embeddings, and saves final ranked file.
    """
    logging.info("--- Starting QA Augmentation Phase ---")

    # 1. Load intermediate Q&A data
    logging.info(f"Loading intermediate Q&A data from {INTERMEDIATE_OUTPUT_FILE}...")
    qa_df = pd.read_excel(INTERMEDIATE_OUTPUT_FILE, engine='openpyxl')
    logging.info(f"Loaded {len(qa_df)} Q&A pairs for augmentation.")

    # 2. Use pre-loaded document data and embeddings array
    all_chunks_text = [item["text"] for item in documents_data_with_embeddings]
    logging.info(f"Using {len(documents_data_with_embeddings)} pre-loaded document chunks and their embeddings.")

    # 3. Normalize embeddings for cosine similarity calculation
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1e-10, norms) # Avoid division by zero
    normalized_embeddings = embeddings_array / safe_norms

    # 4. Find Neighbors based on Source_Chunk similarity
    logging.info(f"Finding top {config.RETRIEVAL_K} semantically similar chunks for each Source_Chunk...")
    results_data = []

    for iter_index, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Augmenting Q&A Pairs"): 
        source_chunk_text = row["Source_Chunk"] 
        question_text = row["Question"] 
        result_row = row.to_dict() 

        # Find the index of the source chunk in our master list by direct comparison
        try:
            source_chunk_index = all_chunks_text.index(source_chunk_text)
        except ValueError:
            # If exact match is not found, log warning and skip
            logging.warning(f"Could not find exact match for source chunk at row {iter_index}. Skipping.")
            continue

        # Get the normalized embedding for the source chunk
        source_embedding = normalized_embeddings[source_chunk_index].reshape(1, -1)

        # Calculate cosine similarity with all normalized embeddings
        # Cosine similarity = dot product of normalized vectors
        similarities = np.dot(normalized_embeddings, source_embedding.T).flatten()

        # Get indices sorted by similarity (highest first)
        # Argsort returns indices that *would* sort the array
        sorted_indices = np.argsort(similarities)[::-1] # Descending order

        # Get the top K indices (should include the source_chunk_index itself at rank 1)
        top_n_indices = sorted_indices[:config.RETRIEVAL_K]

        # Store the text of the ranked chunks
        for i, chunk_idx in enumerate(top_n_indices):
                rank = i + 1
                result_row[f'Relevant_Chunk_Rank_{rank}'] = all_chunks_text[chunk_idx]

        # Fill remaining ranks if less than K were found (e.g., if K > total chunks)
        for i in range(len(top_n_indices) + 1, config.RETRIEVAL_K + 1):
                result_row[f'Relevant_Chunk_Rank_{i}'] = None

        results_data.append(result_row)

    # 6. Create new DataFrame and Save Final Results
    augmented_df = pd.DataFrame(results_data)
    logging.info(f"Saving final augmented Q&A data ({len(augmented_df)} rows) to {FINAL_OUTPUT_FILE}...")
    augmented_df.to_excel(FINAL_OUTPUT_FILE, index=False, engine='openpyxl')
    logging.info(f"Final augmented dataset saved successfully to {FINAL_OUTPUT_FILE}.")
    return True # Signal success

# --- Main Pipeline Execution --- #

def main():
    logging.info("======== Starting Dataset Pipeline ========")

    # --- Initialize Shared Components using rag_core ---
    # This loads model, client, docs and index
    index, documents_data, model_encoder, client = initialize_agent(
        docs_dir=config.DOCS_DIRECTORY,
        index_file=config.INDEX_FILE,
        embedding_model_name=config.EMBEDDING_MODEL,
        chunk_size=config.CHUNK_SIZE
    )


    # --- Run Pipeline Stages ---
    # Pass the necessary components to each stage
    generation_successful, docs_with_clusters_embeddings, embeddings_array = run_qa_generation(client, model_encoder, documents_data)

    # Pass index, model, the processed documents_data, and the embeddings array
    augmentation_successful = run_qa_augmentation(model_encoder, index, docs_with_clusters_embeddings, embeddings_array)

    logging.info("======== Dataset Pipeline Finished Successfully ========")

if __name__ == "__main__":
    main() 