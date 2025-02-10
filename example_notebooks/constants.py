RAG_PROMPT = """You are an AI conversational assistant. Given the following context and question, provide a relevant and accurate answer. If you don't know the answer, just say that you don't know. Use only 50 words or less. \n\n
Be precise and accurate. Make sure you use proper grammar and that you answer the question taking into account the entire context. \n\n
Context:
{context}

Question:
{question}

Answer:"""

GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. \n 
Here is the retrieved document: \n\n {context} \n\n
Here is the user question: {question} \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

AGENT_PROMPT = """
You are an AI assistant. You first need to decide if you need to use the users past interactions. \n
You have the task to generate a new question based on the user's latest question or leave the question as it is. \n
You might need to rewrite the question to make it semantically similar to the user's previous interaction. \n
If the user asks to try again reform the last interaction. \n
Interactions are in the form of question-answer. \n
Here are the user's past interactions:
\n ------- \n
{past_interactions}

Now, the latest question is:
\n ------- \n
{user_question}
"""

REWRITE_QUERY = """ \n 
Look at the input and try to reason about the underlying semantic intent / meaning. \n 
Here is the initial question:
\n ------- \n
{question} 
\n ------- \n
Formulate an improved question: 
Only provide the question. Do not include any other information. \n"""

REWRITE_DECISION = "rewrite"
GENERATE_DECISION = "generate"
