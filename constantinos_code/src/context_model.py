from pydantic import BaseModel
class RagAgentContext(BaseModel):
    """Context for RAG agent.
    
    This context is shared between all agents in the RAG workflow.
    """
    question: str | None = None
    retrieved_documents: list[str] = []
    relevant_documents: list[str] = []
    answer: str | None = None