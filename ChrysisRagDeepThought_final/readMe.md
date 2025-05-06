### 1. Run the Deep Thought RAG Notebook

Open the `chrysisRagDeepThought_colab` notebook and follow its instructions.

Load the `chrysisRagDeepThought_pre-run.zip` folder, prepare a `TOGETHER_API_KEY`, and input it when prompted. Run the code in the specified order.

The file `chrysisRagDeepThought_outputs.zip` will be downloaded after running all cells.

---


### 2.Evaluating Retrieval
When we run the 1_dataset_pipeline.py script, it generates the file generated_qa_dataset_ranked, which is then used by 3_evaluate_retrieval.py to evaluate retrieval performance. However, some of the generated questions are non-specific, leading to poor retrieval. To address this, I manually removed the non-specific questions in a previous run and saved the cleaned output as generated_qa_dataset_ranked_goodQuestions. If you run 3_evaluate_retrieval.py on this file (after renaming it back to generated_qa_dataset_ranked), you’ll see improved metrics.
More specifically, after removing the eight non-specific questions and rerunning retrieval on the remaining 34 high-quality questions (k = 5), the metrics compare as follows:
	• Mean Hit Rate @5: before 0.6905 → now 0.8485
	• Mean Reciprocal Rank @5: before 0.5337 → now 0.6490
	• Mean Precision @5: before 0.3143 → now 0.3879
	• Mean Kendall’s Tau-b @5: before 0.2667 → now 0.3091
These improvements confirm that evaluating only well-posed, specific questions yields more accurate and meaningful retrieval metrics.

### 3. Comparing RAG Performance

If someone wants to compare our RAG, they just need to run their agent through the 42 questions
(the updated version—see file `generated_qa_dataset_ranked.xlsx`)
and compute the Mean Semantic Similarity to the ground truth answers. See the file `4_evaluate_generation.py` for details.

For retrieval, it may be harder to compare our agents, but take a look at the file `3_evaluate_retrieval.py`—you might be able to adapt it to your code.

---

### 4. Run the Preprocessing Notebook

Open the `chrysisRagPrerocess_colab` notebook and follow its instructions.

Load the `chrysisRagPreprocess.zip` folder. Run the code in the specified order.

---
