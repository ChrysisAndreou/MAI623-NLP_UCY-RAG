from agents import function_tool, RunContextWrapper
from .context_model import RagAgentContext
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI
from pydantic import BaseModel

class Relevance(BaseModel):
    is_relevant: bool | None = None

client = OpenAI()

# Define all tools
@function_tool(
    name_override="retrieve_general_documents",
    description_override="Retrieve general documents from the Vector Database."
)
def retrieve_general_documents(wrapper: RunContextWrapper[RagAgentContext], question: str) -> str:
    """Retrieve general documents from the Vector Database.
    
    Args:
        context: The context of the RAG agent
        question: The question to retrieve documents for
    Returns:
        A string indicating the general inquiry documents have been retrieved
    """
    wrapper.context.question = question
    vector_store = FAISS.load_local(
                    "faiss_index",
                    OllamaEmbeddings(model="nomic-embed-text"), 
                    allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(query=wrapper.context.question, k=5)
    wrapper.context.retrieved_documents = [doc.page_content for doc in docs]    
    return f"Retrieved {len(wrapper.context.retrieved_documents)} documents: {wrapper.context.retrieved_documents}"

@function_tool(
    name_override="grade_documents",
    description_override="Grade the relevance of retrieved documents against the user question and filter out irrelevant ones."
)
def grade_documents(wrapper: RunContextWrapper[RagAgentContext]) -> str:
    """Grades retrieved documents for relevance to the question and keeps only relevant ones.

    Args:
        wrapper: The context wrapper containing retrieved documents and the question.

    Returns:
        A string summarizing the filtering results.
    """
    question = wrapper.context.question
    retrieved_documents = wrapper.context.retrieved_documents

    if not retrieved_documents:
        return "No documents to grade."
    if not question:
        return "No question available in context to grade documents against."

    relevant_docs = []
    for doc in retrieved_documents:
        # Invoke the grader
        try:
            response = client.responses.parse(
                        model="gpt-4.1-mini",
                        input=[
                            {"role": "system", "content": "Extract the relevance of the document in relation to the user question."},
                            {
                                "role": "user",
                                "content": f"Document: {doc}\n\nUser question: {question}",
                            },
                        ],
                        text_format=Relevance,
                    )
            grade = response.output_parsed
            if grade.is_relevant == True:
                relevant_docs.append(doc)
        except Exception as e:
            # Handle potential errors during LLM call (e.g., connection issues)
            # Log the error and skip the document for now
            print(f"Error grading document: {e}") # Consider using proper logging

    # Update context with relevant documents
    wrapper.context.relevant_documents = relevant_docs

    return f"Graded documents\n\nFound {len(relevant_docs)}\n\nRelevant documents: {relevant_docs}"