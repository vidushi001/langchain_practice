from langchain_ollama  import ChatOllama
llm = ChatOllama(model="llama3.2")
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel,Field
#simple example
class User(BaseModel):
    name:str
    age:int = 32
    weight:Optional[str]=None

reviewModel= llm.with_structured_output(User)
prompt = "hi i am vicky tyagi and my age is 31 my weight is 70.8kg"
response = reviewModel.invoke(prompt)
print(response)

#Field example
class Review(BaseModel):
    key_output:list[str]=Field(description="Write down all the key themes discussed in the review in a list")
    Sentiment:Literal['positive','negative']=Field(description="Return sentiment of the review either negative, positive or neutral")
    name:Optional[str] = Field(description="Name of reviewer",default=None)
    pros:Optional[list[str]] = Field(description="Write down all the pros")
    cons:Optional[list[str]] = Field(description="Write down all the cons")
    StarValue:Optional[int] = Field(gt = 0, lt = 6,desciption = "Star value mention in review,if start value does not matched then consider 5")

reviewModel= llm.with_structured_output(Review)    
userReview = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
I will give 2 start to this product
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""
response2 = reviewModel.invoke(userReview)
print(response2)
# json_res = response2.model_dump_json()
# print(json_res.StarValue)
# print(json_res.Sentiment)
# print(json_res.name)
# print(json_res.key_output)


