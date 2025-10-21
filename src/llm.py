from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

# Setup the LLM
def get_llm():
    """Get the LLM for the langgraph agent."""
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.0,
        top_p=1.0,
    )
    return llm