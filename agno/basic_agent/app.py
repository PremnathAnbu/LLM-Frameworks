from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = (
    os.getenv("GROQ_API_KEY")
    .strip()
    .replace('"', "")
    .replace("'", "")
)

print(repr(api_key))

agent = Agent(
    model=Groq(
        id="llama-3.3-70b-versatile",
        api_key=api_key
    ),
    markdown=True
)

agent.print_response("Tell me the today IPL match")