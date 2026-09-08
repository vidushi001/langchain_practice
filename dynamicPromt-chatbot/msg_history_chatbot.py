from langchain_ollama  import ChatOllama;
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
llm = ChatOllama(model="llama3.2")
chat_history=[SystemMessage(content='You are a helpful AI assistant')]

while True:
    userInput = input('You: ')
    chat_history.append(HumanMessage(content=userInput))
    if userInput == 'exit':
        break
    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(result.content)

print(chat_history)    