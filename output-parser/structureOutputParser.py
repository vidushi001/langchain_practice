from langchain_ollama  import ChatOllama
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2")

schema = [
    ResponseSchema(name='name', description='name about the topic'),
    ResponseSchema(name='age', description='age about the topic'),
    ResponseSchema(name='weight', description='weight about the topic'),
    ResponseSchema(name='occupation', description='occupation about the topic'),
    ResponseSchema(name='study', description='study about the topic'),
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
    
)
chain = template | llm | parser

result = chain.invoke({'topic':'Hi my name is vidushi tyagi.My age is 32 and my weight is 54.I am science student.I am working'})

print(result)