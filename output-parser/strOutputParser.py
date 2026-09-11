from langchain_ollama  import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2")

parser = StrOutputParser()
template = PromptTemplate(
    template='Write 5 fact about {topic}',
    input_variables=['topic']
)
chain = template | llm | parser

result = chain.invoke({'topic':'india'})

print(result)