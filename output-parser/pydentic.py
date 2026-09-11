from langchain_ollama  import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2")

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    study: str = Field(description='Name of study') 
    weight: int = Field(gt=18, description='weight of the person')
    occupation: str = Field(description='occupation of the person') 

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
    
)
chain = template | llm | parser

result = chain.invoke({'topic':'Hi my name is vidushi tyagi.My age is 54 and my weight is 54.I am science student.I am working as IT engineer'})

print(result)