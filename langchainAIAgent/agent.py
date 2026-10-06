from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
import requests


search_tool = DuckDuckGoSearchRun()

prompt = """
You are a helpful assistant.
Use the available tools whenever required.
"""

llm = ChatOllama(
    model="llama3.2",
    temperature=1.5
)


@tool
def get_weather(city: str) -> str:
    """Get the current weather of the given city."""

    url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

    response = requests.get(url)
    return response.text


agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather],
    system_prompt=prompt
)


query = "Please give me the current temperature of the capital of India."

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": query
        }
    ]
})

print(response["messages"][-1].content)