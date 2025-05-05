from agents import Agent

from .context_model import RagAgentContext
from .prompts import (
    TRIAGE_PROMPT,
    GRADER_PROMPT,
    RETRIEVAL_PROMPT,
)
from .tools import (
    retrieve_general_documents,
    grade_documents,
)

from dotenv import load_dotenv
import os
load_dotenv()

# Define all agents
grader_agent = Agent[RagAgentContext](
    model=os.getenv("MODEL"),
    name="Grader Agent",
    handoff_description="An expert agent that can grade the quality of the retrieved documents in relation to the user's question.",
    instructions=GRADER_PROMPT,
    tools=[grade_documents],
)

# Retrieval Agent
retrieval_agent = Agent[RagAgentContext](
    model=os.getenv("MODEL"),
    name="Retrieval Agent",
    handoff_description="An expert agent that can retrieve documents from the Vector Database.",
    instructions=RETRIEVAL_PROMPT,
    tools=[retrieve_general_documents],
    handoffs=[grader_agent],
)

triage_agent = Agent[RagAgentContext](
    model=os.getenv("MODEL"),
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a user's request to the appropriate agent.",
    instructions=TRIAGE_PROMPT,
    handoffs=[
        retrieval_agent,
        grader_agent,
    ],
)

# Allow sub-agents to return to triage if needed (human review loop)
retrieval_agent.handoffs.append(triage_agent)
grader_agent.handoffs.append(triage_agent)

starting_agent = triage_agent
