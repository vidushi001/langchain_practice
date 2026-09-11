from langchain_ollama  import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

parser = StrOutputParser()
model = ChatOllama(model="llama3.2")

promt1 = PromptTemplate(
    template="give 5 pros about {topic}",
    input_variables=['topic']
)

promt2 = PromptTemplate(
    template="give 5 cons about {topic}",
    input_variables=['topic']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n Pros -> {Pros} and cons -> {cons}',
    input_variables=['Pros', 'cons']
)

parellelChain = RunnableParallel({
    'Pros': promt1 | model | parser,
    'cons': promt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = parellelChain | merge_chain

output = chain.invoke({'topic':'democracy'})

print(output)