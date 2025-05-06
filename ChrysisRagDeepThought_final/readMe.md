### 1. Run the Deep Thought RAG Notebook

Open the `chrysisRagDeepThought_colab` notebook and follow its instructions.

Load the `chrysisRagDeepThought_pre-run.zip` folder, prepare a `TOGETHER_API_KEY`, and input it when prompted. Run the code in the specified order.

The file `chrysisRagDeepThought_outputs.zip` will be downloaded after running all cells.

---

### 2. Comparing RAG Performance

If someone wants to compare our RAG, they just need to run their agent through the 42 questions
(the updated version—see file `generated_qa_dataset_ranked.xlsx`)
and compute the Mean Semantic Similarity to the ground truth answers. See the file `4_evaluate_generation.py` for details.

For retrieval, it may be harder to compare our agents, but take a look at the file `3_evaluate_retrieval.py`—you might be able to adapt it to your code.

---

### 3. Run the Preprocessing Notebook

Open the `chrysisRagPrerocess_colab` notebook and follow its instructions.

Load the `chrysisRagPreprocess.zip` folder. Run the code in the specified order.

---
