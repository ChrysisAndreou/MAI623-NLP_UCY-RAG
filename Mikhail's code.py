

import os
import streamlit as st
import time
import pandas as pd
import json
import torch
import pickle
import langchain.agents as agents
import pypdf
import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from transformers import AutoTokenizer, AutoModel
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal
from langgraph.graph import START
from langgraph.graph import StateGraph, START, END
from langchain.document_loaders import JSONLoader
from langgraph.graph.message import add_messages
from langchain_community.embeddings.voyageai import VoyageEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.vectorstores import Pinecone
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from operator import itemgetter
from langchain.load import dumps, loads
from typing_extensions import TypedDict
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from typing import Annotated
from langchain import hub
from langchain_community.tools import TavilySearchResults
from typing import Annotated
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain import OpenAI, PromptTemplate, VectorDBQA, LLMChain
from langchain.agents import Tool, initialize_agent, AgentExecutor, AgentOutputParser, LLMSingleActionAgent, create_react_agent, initialize_agent, load_tools, create_tool_calling_agent, AgentExecutor
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, CombinedMemory
from langchain.prompts import StringPromptTemplate
from langchain.schema import HumanMessage, AgentAction, AgentFinish
from langchain.text_splitter import CharacterTextSplitter
from langchain import hub
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, SystemMessage

##################### SUPPRESS LANGCHAIN WARNINGS ###############
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*LangChainBetaWarning.*")
##################### SUPPRESS LANGCHAIN WARNINGS ###############


os.environ['OPENAI_API_KEY'] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["PINECONE_API_KEY"] = ""
os.environ["VOYAGE_API_KEY"] = ""

embeddings = VoyageAIEmbeddings(
    model="", 
    output_dimension=2048 
)

vectorstore_general_url = PineconeVectorStore(
    index_name="", 
    namespace="",
    embedding=embeddings
)

vectorstore_general_general = PineconeVectorStore(
    index_name="", 
    namespace="",
    embedding=embeddings
)

vectorstore_schedule = PineconeVectorStore(
    index_name="", 
    embedding=embeddings
)

vectorstore_formfetcher = PineconeVectorStore(
    index_name="",
    namespace="", 
    embedding=embeddings
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.4)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    sys_msg: str


@tool
def fillout_form_fetcher(user_question):
    """This node is needed for fetching URLs for forms to fill out. Usually has to be used in conjunction with the general node or by itself"""
    form_retriever = vectorstore_formfetcher.as_retriever(search_kwargs={"k": 3})

    form_prompt = ChatPromptTemplate.from_template("""This is a University Fill-out Form fetcher. I am going to give you a user question and
                                                    some JSON objects that contain the following:
    1) The name of a document
    2) The URL of the document
    3) The description of a document. 
    Your task is to return an appropriate form and its URL based on the user question I'll provide below. Don't double
    the URL and remove any casing so as not to break the link. Don't give me a lot of narration, but do summarize the description
    for these forms. Your answer should be concise and to the point. Do not double the link, e.g. this - Form URL: []
    should be parsed into two separate links. Be polite.
    User question: {question}
    Jsons that were retrieved: {jsons}""")
    chain = {"jsons": form_retriever, "question": RunnablePassthrough()} | form_prompt | llm | StrOutputParser()
    response_acquired = chain.invoke(str(user_question))
    return {"response": str(response_acquired)}

@tool
def schedule_question(user_question):
    """For questions regarding schedules, lesson rooms (where a lesson is conducted - navigating the user to a room is another node's responsibility), buildings, professors. After the answer, ask if the user would like to be directed to the building / room. If the question is predominantly in Greek, then translate it to Greek fully. If it is predominantly in English, translate it to English fully."""
    question = str(user_question)
    retriever_schedule = vectorstore_schedule.as_retriever(search_kwargs={"k": 12})
    template = ChatPromptTemplate.from_template("""This is a node within a student RAG system. 
    This particular node is responsible for providing answers regarding schedules for students. These
    shedules are in the form of json objects and mention multiple aspects like: CRN (course number), Course ,Term,
    ,Course, Title, ECTS points for the course, MTWRFS (day of the week, R is thursday), Time, Building, Room, Room Cap(acity),
    number of Seats, Instructor Name. Sometimes some of these aspects are not mentioned. Some building names may accidentally 
    be distorted by the OCR system, so please ensure they're written properly: ΘΕΕ01, ΘΕΕ02, ΟΕΔ02, ΚΕΠ, ΧΩΔ02, ΧΩΔ01, ΠΤΕΡ, ΗΛΙΑΔΗ
    I need you to provide me an answer based on the context I am going to provide. Do not hallucinate and avoid
    giving wrong information. If you are not sure about the answer, please say so. If there are multiple subjects with some name, return all of them
    The answer should be concise and to the point. After the answer, ask if the user would like to be directed to the building / room
    Context: {context}
    Question: {question}""")
    chain = {"context": retriever_schedule, "question": RunnablePassthrough()} | template | llm | StrOutputParser()
    response_acquired = chain.invoke(str(question))
    return {"response": str(response_acquired)}

@tool   
def general_question(user_question):
    """For general questions about the university, courses, teachers, programmes, etc."""
    uquest = str(user_question)
    
    retriever_general_general = vectorstore_general_general.as_retriever(search_kwargs={"k": 3})
    retriever_general_urls = vectorstore_general_url.as_retriever(search_kwargs={"k": 3})

    template = """I am going to give you a user question and some context that was retrieved from a university documentation. Some of these will be URLs and some of those will be regular documents. Answer
    the question based on the received context. There will be two retrievers, and the second one will involve precise URLs that, if relevant, must be provided. The answer should complete and involve all the information requested, can't skip anything. If you don't know the answer, please just say so. Provide links to all the URL pages that you got the information from. 
    If a question concerns courses, try to optionally find a brochure for a course and provide a link for it. If there is conflicting information, ensure that WEBSITE URL INFO holds the priority (ANSWER IN A DETAILED MANNER):

    General Info (do not show urls for this part - this is just general information):
    {context1}

    With URLs (provide URLs for this information - if the same link repeats, don't show it twice - only where necessary should you do it):
    {context2}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        {"context1": retriever_general_general, "context2": retriever_general_urls,
            "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    response_acquired = final_rag_chain.invoke({"question":str(user_question)})
    return {"response": str(response_acquired)}

### System Message - Gets appended at the beginning of each conversation cycle (including summaries)
sys_msg = SystemMessage(content="""You are a university RAG chatbot for the University of Cyprus. You are
                        helping students get information based on real university documents. 
                        You have 3 tools:
                        - general_question - for general questions about the university, teachers, courses, etc. ;
                        - schedule_question - for precise questions about the time / location / instructor of lessons ;
                        - fillout_form_fetcher - when a user asks for a form to fill out ;
                        Answer a question using some of the tools.""")

def chatbot(state: State):
    messages = state["messages"]
    if sys_msg not in messages:  
        messages = [sys_msg] + messages
    return {"messages": [llm_with_tools.invoke(messages)]}

graph_builder = StateGraph(State)
tools = [general_question, schedule_question, fillout_form_fetcher]
multiple_tools = ToolNode([general_question, schedule_question, fillout_form_fetcher])
llm_with_tools = llm.bind_tools(tools)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_node("tools", multiple_tools)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")

graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()

config = {"configurable": {"thread_id": "1"}}

graph = graph_builder.compile(checkpointer=memory)

user_input = str(input("Enter your question: "))
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",

for event in events:
    latest_message = event["messages"][-1]

assistant_reply = latest_message


