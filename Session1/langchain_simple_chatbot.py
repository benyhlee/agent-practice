# pip install langchain langchain-openai

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage("너는 사용자를 도와주는 친절한 조력자야.")
]

while True:
    user_input = input("사용자: ") 

    if user_input == "exit":
        break
    
    messages.append(
        HumanMessage(user_input)
    )
    
    response = llm.invoke(messages)
    print(response)
    messages.append(response) # AIMessage()로 추가됨
    print("AI: " + response.content)



