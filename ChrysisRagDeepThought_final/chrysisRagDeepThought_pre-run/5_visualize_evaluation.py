import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# --- Configuration ---
GENERATION_RESULTS_FILE = "generation_evaluation_results.csv"
RETRIEVAL_SUMMARY_FILE = "retrieval_evaluation_summary.json"
OUTPUT_DIR = "evaluation_plots"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logging.info(f"Created output directory: {OUTPUT_DIR}")

def plot_semantic_similarity(results_df):
    """Generates and saves a histogram/KDE plot for semantic similarity."""

    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['semantic_similarity'], kde=True, bins=20)
    plt.title('Distribution of Semantic Similarity Scores')
    plt.xlabel('Semantic Similarity (Cosine)')
    plt.ylabel('Frequency')
    mean_sim = results_df['semantic_similarity'].mean()
    plt.axvline(mean_sim, color='r', linestyle='--', label=f'Mean: {mean_sim:.3f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plot_filename = os.path.join(OUTPUT_DIR, "semantic_similarity_distribution.png")
    try:
        plt.savefig(plot_filename)
        logging.info(f"Saved semantic similarity plot to {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to save plot {plot_filename}: {e}")
    plt.close()

def plot_retrieval_metrics(summary_data):
    """Generates and saves bar charts for retrieval metrics across K values."""
    try:
        # Convert summary data keys (K values as strings) to integers for sorting
        k_values = sorted([int(k) for k in summary_data.keys()])
        
        # Dynamically determine base metric names (e.g., 'Mean Hit Rate', 'MRR')
        # Assumes all K values have the same set of metrics, just with different @k suffix
        first_k_metrics = summary_data[str(k_values[0])].keys()
        # Extract base names by splitting at the last '@'
        base_metric_names = sorted(list(set(m.rsplit('@', 1)[0].strip() for m in first_k_metrics)))

        # Prepare data for plotting using base names
        plot_data = {base_name: [] for base_name in base_metric_names}
        for k in k_values:
            k_str = str(k)
            for base_name in base_metric_names:
                # Construct the full metric name for the current K
                metric_key = f"{base_name} @{k}"
                # Append the value, using get() with a default of NaN if a metric is unexpectedly missing
                plot_data[base_name].append(summary_data[k_str].get(metric_key, float('nan')))

        # Use base_metric_names for the DataFrame columns
        df_plot = pd.DataFrame(plot_data, index=k_values)
        df_plot.index.name = 'K'

        # Plotting
        num_metrics = len(base_metric_names)
        # Create a 2x2 grid for the plots, assuming num_metrics is 4
        # Adjust figsize for a more square layout
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True)
        fig.suptitle('Retrieval Evaluation Metrics vs. K', fontsize=16, y=1.02) # Adjust title position
        
        # Flatten axes array for easier iteration if needed, or use 2D indexing
        axes_flat = axes.flatten()

        for i, base_name in enumerate(base_metric_names):
            # Determine row and column index for the 2x2 grid
            row, col = divmod(i, 2) # divmod(i, 2) gives (i // 2, i % 2)
            ax = axes[row, col]
            
            df_plot[base_name].plot(kind='bar', ax=ax, rot=0)
            # Use the base name for the title
            ax.set_title(base_name.replace("Mean ", "").replace(" Tau-b", "")) # Cleaner titles
            ax.set_ylabel('Score')
            ax.grid(axis='y', alpha=0.5)

            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f')
                
            # Set x-label only for the bottom row plots
            if row == 1:
                 ax.set_xlabel('K (Number of Retrieved Documents)')
            else:
                 # Remove x-tick labels for top row plots to avoid overlap
                 ax.tick_params(axis='x', labelbottom=False)

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        plot_filename = os.path.join(OUTPUT_DIR, "retrieval_metrics_comparison.png")
        plt.savefig(plot_filename)
        logging.info(f"Saved retrieval metrics plot to {plot_filename}")
        plt.close()

    except Exception as e:
        logging.error(f"Failed to process or plot retrieval metrics: {e}")

def main():
    logging.info("--- Starting Evaluation Visualization ---")

    # --- Load Generation Results ---
    if os.path.exists(GENERATION_RESULTS_FILE):
        logging.info(f"Loading generation results from: {GENERATION_RESULTS_FILE}")
        try:
            gen_results_df = pd.read_csv(GENERATION_RESULTS_FILE)
            logging.info(f"Loaded {len(gen_results_df)} generation results.")
            plot_semantic_similarity(gen_results_df)
        except Exception as e:
            logging.error(f"Failed to load or process {GENERATION_RESULTS_FILE}: {e}")
    else:
        logging.warning(f"Generation results file not found: {GENERATION_RESULTS_FILE}. Skipping generation plots.")

    # --- Load Retrieval Summary ---
    if os.path.exists(RETRIEVAL_SUMMARY_FILE):
        logging.info(f"Loading retrieval summary from: {RETRIEVAL_SUMMARY_FILE}")
        try:
            with open(RETRIEVAL_SUMMARY_FILE, 'r') as f:
                retrieval_summary = json.load(f)
            logging.info("Loaded retrieval summary.")
            plot_retrieval_metrics(retrieval_summary)
        except Exception as e:
            logging.error(f"Failed to load or process {RETRIEVAL_SUMMARY_FILE}: {e}")
    else:
        logging.warning(f"Retrieval summary file not found: {RETRIEVAL_SUMMARY_FILE}. Skipping retrieval plots.")

    logging.info("--- Evaluation Visualization Finished ---")

if __name__ == "__main__":
    main() 