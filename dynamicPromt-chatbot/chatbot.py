from langchain_ollama  import ChatOllama;
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2", temperature=1.5)

template2 = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',
    input_variables=['name']
)
prompt = template2.invoke({'name':'nitish'})

result = llm.invoke(prompt)

print(result.content)



