from agno.agent import Agent
from agno.models.groq import Groq
from sentence_transformers import SentenceTransformer
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools

from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import PyPDFLoader
from agno.vectordb.lancedb import LanceDb,SearchType
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
import os
from dotenv import load_dotenv
load_dotenv()
api_key = (
    os.getenv("GROQ_API_KEY")
    .strip()
    .replace('"', "")
    .replace("'", "")
)
model = SentenceTransformer(
    "BAAI/bge-m3"
)
# print(repr(api_key))

agent = Agent(
    model=Groq(
        id="llama-3.3-70b-versatile",
        api_key=api_key
    ),
    description="You are a Thai cuisine expert!",
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results."
    ],
    knowledge=PDFUrlKnowledgeBase(urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
                                  vector_db=LanceDb(
                                      uri="tmp/lancedb",
                                      table_name="recipes",
                                      search_type=SearchType.hybrid,
                                      embedder=SentenceTransformerEmbedder(
                id="BAAI/bge-m3"
            )
                                  ),),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

if agent.knowledge is not None:
    agent.knowledge.load()
    
agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
agent.print_response("What is the history of Thai curry?", stream=True)
