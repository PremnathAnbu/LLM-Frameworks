from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv
load_dotenv()

api_key = (
    os.getenv("GROQ_API_KEY")
    .strip()
    .replace('"', "")
    .replace("'", "")
)

web_agent=Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile",
    api_key=api_key),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    markdown=True,
    
)
finance_agent=Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-versatile",
    api_key=api_key),
    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True)],
    instructions="Use tables to display data",
    markdown=True,
)

agent_team=Agent(
    team=[web_agent,finance_agent],
    model=Groq(id="llama-3.3-70b-versatile",
    api_key=api_key),
    instructions=["Always include sources","Use tables to display data"],
    markdown=True,
)

agent_team.print_response("analyze the market outlook and financial performance of AI semiconductor company NVDA and Tesla And suggest whether I have to buy or not?")