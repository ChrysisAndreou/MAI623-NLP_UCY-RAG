# Chrysis RAG DeepThought: A Conversational AI for University of Cyprus Information

## Overview

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system designed to function as a conversational AI expert on the University of Cyprus. The system, named "DeepThought," leverages information scraped from the university's official website to provide accurate, context-aware answers to user queries.

The project is structured into three main components:
1.  **Data Preprocessing**: A pipeline that automatically gathers, cleans, and prepares the source material from the web.
2.  **RAG System & Evaluation Suite**: The core of the project, which includes an automated pipeline for generating a high-quality Question-Answer dataset for evaluation, an interactive chat agent, and a comprehensive suite of scripts to measure the system's performance.
3.  **Interactive Agent**: A user-friendly command-line interface that allows users to directly interact with the "DeepThought" AI.

## Directory Structure

The project is organized into the following directories:

```
└── ChrysisRagDeepThought_final/
    ├── readMe.md
    ├── chrysisRagPreprocess_colab.ipynb
    ├── chrysisRagDeepThought_pre-run/
    │   ├── 1_dataset_pipeline.py
    │   ├── 2_interactive_agent.py
    │   ├── 3_evaluate_retrieval.py
    │   ├── 4_evaluate_generation.py
    │   ├── 5_visualize_evaluation.py
    │   ├── config.py
    │   ├── generated_qa_dataset_ranked_qoodQuestions.xlsx
    │   ├── rag_core.py
    │   ├── requirements.txt
    │   └── utils.py
    ├── chrysisRagPreprocess/
    │   ├── 01_links.csv
    │   ├── 01_linksToHtml.py
    │   ├── 02_htmlToMD.py
    │   ├── 03_clean_md.py
    │   ├── 04_long_files.py
    │   └── requirements.txt
    └── output_for automatically generate questions/
        └── retrieval_evaluation_summary.json
    └── output_for good questions/
        └── retrieval_evaluation_summary.json
```

*   `chrysisRagPreprocess/`: Contains all scripts for the data preprocessing pipeline.
*   `chrysisRagDeepThought_pre-run/`: Houses the core RAG system, evaluation scripts, and the final data (`long_files/`).
*   `output_.../`: Stores the JSON results from the evaluation runs.
*   `*_colab.ipynb`: Jupyter notebooks that provide a guided way to execute the preprocessing and RAG pipelines.

## Core Components Explained

### 1. Data Preprocessing (`chrysisRagPreprocess/`)

This module is responsible for building the knowledge base for the RAG agent. The process is as follows:
1.  **Link Collection**: A list of URLs from the University of Cyprus website is defined in `01_links.csv`.
2.  **HTML Scraping**: `01_linksToHtml.py` fetches the raw HTML content from these URLs.
3.  **Markdown Conversion**: `02_htmlToMD.py` converts the HTML files into a cleaner, more structured Markdown format.
4.  **Content Cleaning**: `03_clean_md.py` intelligently removes irrelevant clutter like navigation bars, headers, and footers, isolating the main informational content.
5.  **Filtering**: `04_long_files.py` discards documents with minimal content, ensuring that only information-rich files are used by the RAG system. The final dataset resides in `chrysisRagDeepThought_pre-run/long_files`.

### 2. The RAG System & Evaluation Suite (`chrysisRagDeepThought_pre-run/`)

This is the heart of the project, containing the logic for the AI and its evaluation.

*   **Automated QA Dataset Generation (`1_dataset_pipeline.py`)**: A key innovation of this project is its ability to create its own high-quality evaluation dataset. It processes the cleaned text, clusters it to identify diverse topics, and then uses a Large Language Model (LLM) to generate relevant question-answer pairs from representative text chunks. This process results in the `generated_qa_dataset_ranked_qoodQuestions.xlsx` file, which serves as the "ground truth" for evaluations.

*   **Interactive Chat Agent (`2_interactive_agent.py`)**: This script launches a command-line interface that allows you to chat directly with "DeepThought." It takes your question, retrieves relevant documents from its knowledge base, and generates a coherent answer.

*   **Evaluation Suite (`3_evaluate_retrieval.py`, `4_evaluate_generation.py`, `5_visualize_evaluation.py`)**: The project includes a robust framework to rigorously assess its performance.
    *   `3_evaluate_retrieval.py`: Measures the effectiveness of the retrieval component using metrics like **Hit Rate**, **Mean Reciprocal Rank (MRR)**, **Precision**, and **Kendall's Tau**.
    *   `4_evaluate_generation.py`: Evaluates the quality of the final generated answers by comparing them to the ground truth answers using **Semantic (Cosine) Similarity**.
    *   `5_visualize_evaluation.py`: Creates insightful plots and charts from the evaluation results, making it easy to visualize the system's performance.

## How to Run

### Prerequisites

1.  **Python Environment**: Ensure you have Python installed.
2.  **API Key**: The system uses the Together AI platform to access powerful Large Language Models.
    *   Sign up for a free account at [Together.ai](https://together.ai) and obtain an API key.
    *   Set this key as an environment variable.
        ```bash
        export TOGETHER_API_KEY='your_api_key_here'
        ```
3.  **Dependencies**: Each main folder (`chrysisRagPreprocess` and `chrysisRagDeepThought_pre-run`) has its own `requirements.txt` file.

### Option 1: Run the Interactive RAG Agent

This is the quickest way to interact with the "DeepThought" AI.

1.  **Navigate to the Directory**:
    ```bash
    cd ChrysisRagDeepThought_final/chrysisRagDeepThought_pre-run/
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set API Key**: Make sure your `TOGETHER_API_KEY` is set as an environment variable as described in the prerequisites.
4.  **Run the Agent**:
    ```bash
    python 2_interactive_agent.py
    ```
    You will be greeted by the DeepThought persona and can start asking questions about the University of Cyprus.

### Option 2: Execute the Full Pipeline (Replication & Evaluation)

This option is for users who wish to replicate the entire project, from data collection to evaluation. You can use the provided Colab notebooks for a step-by-step guided execution or run the scripts manually.

**A. Data Preprocessing**
*   **Method**: Open the `chrysisRagPreprocess_colab.ipynb` notebook and run the cells in order.
*   **Note**: The first script (`01_linksToHtml.py`) requires credentials for the BrightData service to scrape websites. However, since the HTML and subsequent files are already provided, **you can skip running this specific script** and use the existing data.

**B. Dataset Generation and Evaluation**
*   **Method**: Open the `chrysisRagDeepThought_colab.ipynb` notebook and follow its instructions to run the entire RAG pipeline.
*   **Manual Execution**: Alternatively, from the `chrysisRagDeepThought_pre-run/` directory, run the scripts in the following order after installing requirements and setting the API key:
    1.  `python 1_dataset_pipeline.py` - To generate the QA dataset.
    2.  `python 3_evaluate_retrieval.py` - To evaluate the retrieval component.
    3.  `python 4_evaluate_generation.py` - To evaluate the generation component.
    4.  `python 5_visualize_evaluation.py` - To generate performance plots.

## Comparing RAG Performance

This project's evaluation framework can be used to benchmark other RAG systems.

*   **Evaluating Generation Quality**: To compare your agent, use the `generated_qa_dataset_ranked_qoodQuestions.xlsx` file. For each "Question," generate an answer using your system. Then, calculate the mean semantic similarity between your generated answers and the "Answer" (ground truth) column in the spreadsheet. The `4_evaluate_generation.py` script serves as a reference for this calculation.

*   **Evaluating Retrieval Quality**: Comparing retrieval can be more complex. However, you can adapt the `3_evaluate_retrieval.py` script. The goal is to check if your retrieval system can find the "Relevant_Chunk_Rank_1" (the original source chunk) within its top retrieved documents for each question. This will allow you to calculate comparable metrics like Hit Rate and MRR.
