from langchain_ollama  import ChatOllama
llm = ChatOllama(model="llama3.2")
from typing import TypedDict, Annotated, Optional, Literal
#simple example
Json_schema = {
    "title": "userData",
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "age": {
            "default": 32,
            "type": "integer"
        },
        "weight":{
            "default":None,
            "type":"number",
            "description":"Extract weight from input"

        }
    },
    "required": ["name"]
}

reviewModel= llm.with_structured_output(Json_schema)
prompt = "hi i am vidushi tyagi and my age is 32 and I am  56kg"
response = reviewModel.invoke(prompt)
print(response)

#annotated example
# class Review(TypedDict):
#     key_output:Annotated[list[str],"Write down all the key themes discussed in the review in a list"]
#     Sentiment:Annotated[Literal['positive','negative'],"Return sentiment of the review either negative, positive or neutral"]
#     name:Annotated[Optional[str],"Name of reviewer"]
#     pros:Annotated[Optional[list[str]],"Write down all the pros"]
#     cons:Annotated[Optional[list[str]],"Write down all the cons"]

# reviewModel= llm.with_structured_output(Review)    
# userReview = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

# The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

# However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

# Pros:
# Insanely powerful processor (great for gaming and productivity)
# Stunning 200MP camera with incredible zoom capabilities
# Long battery life with fast charging
# S-Pen support is unique and useful
                                 
# Review by Nitish Singh
# """
# response2 = reviewModel.invoke(userReview)
# print(response2['Sentiment'])


