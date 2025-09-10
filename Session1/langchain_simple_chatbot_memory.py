from langchain_core.chat_history import InMemoryChatMessageHistory  # 대화를 기록하는 클래스
from langchain_core.runnables.history import RunnableWithMessageHistory  # 대화 기록을 활용하는 wrapper
from langchain_openai import ChatOpenAI 
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

# 대화를 저장할 딕셔너리
store = {}

# 대화 기록을 가져오는 함수(세션 ID 필요)
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory() # session_id를 키로해서 메모리 객체 생성
    return store[session_id]

# 대화 기록을 활용하는 모델을 랩퍼로 생성
llm_with_memory = RunnableWithMessageHistory(llm, get_session_history)

# config에서 세션 ID를 설정
config = {"configurable": {"session_id": "1234"}}

# 초기 시스템 메시지를 메모리에 추가
system_message = SystemMessage(content="너는 사용자를 도와주는 친절한 조력자야.")
store["1234"] = InMemoryChatMessageHistory()
store["1234"].add_message(system_message)

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break
    
    human_message = HumanMessage(content=user_input)

    # 답변을 한꺼번에 받을 때
    response = llm_with_memory.invoke([human_message], config=config)
    print(f"AI : {response.content}")

    # 긴 답변을 스트리밍으로 받을 때
    # print("AI: ", end="")
    # for r in llm_with_memory.stream([human_message], config=config):
    #     print(r.content, end="")
    # print("\n")
  
