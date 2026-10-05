from langchain_ollama  import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
#there are 4 step
# 1.tool creation
@tool
def sum(a:int,b:int) -> int:
    """Given input number will be sum and return the output"""
    return a + b

@tool
def multiply(a:int,b:int) -> int:
    """Given input number will be multiply and return the output"""
    return a * b    
# 2.tool calling
# print(sum.invoke({'a':10,'b':30}))    
# print(sum.name)
# print(sum.description)
# print(sum.args)
tools = [sum, multiply]

# Create dynamic tool lookup
tools_map = {
    tool.name: tool
    for tool in tools
}

message=[]
llm = ChatOllama(model="llama3.2", temperature=1.5)

query = 'Calculate both the sum and multiplication of 100 and 55.'
message.append(HumanMessage(content=query))

# 3.tool binding
llm_tools = llm.bind_tools([sum,multiply])

tools_output = llm_tools.invoke(query)
message.append(tools_output)
for toolcall in tools_output.tool_calls:
    tool_name = toolcall['name']
    tool_args = toolcall['args']
# 4.tool execution
    selected_tool = tools_map[tool_name]
    toolMsg = selected_tool.invoke(toolcall)
    message.append(toolMsg)

print("message 49",message)
output = llm_tools.invoke(message)
print(output.content)
