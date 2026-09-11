from langchain_ollama  import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2")

parser = JsonOutputParser()
template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
    
)
chain = template | llm | parser

result = chain.invoke({'topic':'Hi my name is vidushi tyagi.My age is 32 and my weight is 54.I am science student.I am working'})

print(result)