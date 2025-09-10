# tools_1.py
from datetime import datetime
from langchain_core.tools import tool
import pytz
from pydantic import BaseModel, Field
import yfinance as yf
from io import BytesIO
import base64
import matplotlib.pyplot as plt

@tool
def get_current_time(timezone: str, location: str) -> str:
    """ 현재 시각을 반환하는 함수

    Args:
        timezone (str): 타임존(예: 'Asia/Seoul'). 실제 존재해야 함.
        location (str): 지역명. 타임존은 모든 지명에 대응되지 않으므로 이후 llm 답변 생성에 사용됨.
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    location_and_local_time = f'{timezone} ({location}) 현재 시각 {now} '
    print(location_and_local_time)
    return location_and_local_time

class StockHistoryInput(BaseModel):
    ticker: str = Field(..., title="주식 코드", description="주식 코드 (예: AAPL)")
    period: str = Field(..., title="기간", description="주식 데이터 조회 기간 (예: 1d, 1mo, 1y)")

@tool
def get_yf_stock_history(stock_history_input: StockHistoryInput) -> str:
    """ 주식 종목의 가격 데이터를 조회하는 함수"""
    stock = yf.Ticker(stock_history_input.ticker)
    history = stock.history(period=stock_history_input.period)
    history_md = history.to_markdown() 
    return history_md

@tool
def get_yf_stock_info(ticker: str) -> str:
    """
    해당 종목의 Yahoo Finance 정보를 반환합니다.

    Args:
        ticker (str): 정보를 조회하려는 주식 종목의 코드   
    """
    stock = yf.Ticker(ticker)
    info = stock.info # dict
    print(info)
    return str(info)

@tool
def get_yf_stock_recommendations(ticker: str):
    """
    해당 종목의 Yahoo Finance 매수 추천 정보를 반환합니다.

    Args:
        ticker (str): 추천 정보를 조회하려는 주식 종목의 코드   
    """    
    stock = yf.Ticker(ticker)
    recommendations = stock.recommendations # pandas DataFrame
    recommendations_md = recommendations.to_markdown()
    print(recommendations_md)
    return recommendations_md

@tool
def get_stock_chart(ticker: str, start_date: str, end_date: str) -> str:

    """
    해당 종목의 주가 차트를 반환합니다.

    Args:
        ticker (str): 주가 정보를 조회하려는 주식 종목의 코드 (예: AAPL)
        start_date (str): 데이트의 시작일 (예: '2020-05-21')
        end_date (str): 데이터의 종료일 (예: '2023-10-30')
    """ 

    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError("데이터가 없습니다. 종목 코드를 확인해주세요.")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Close'])
    ax.set_title(f"{ticker.upper()} Price ({start_date} ~ {end_date})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.5)
    # ax.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # base64로 인코딩하여 전달
    return base64.b64encode(buf.read()).decode()

# 모든 도구를 리스트로 내보내기
ALL_TOOLS = [get_current_time, get_yf_stock_history, get_yf_stock_info, get_yf_stock_recommendations, get_stock_chart]

# 도구 딕셔너리도 내보내기
TOOL_DICT = {
    "get_current_time": get_current_time, 
    "get_yf_stock_history": get_yf_stock_history,
    "get_yf_stock_info": get_yf_stock_info,
    "get_yf_stock_recommendations": get_yf_stock_recommendations,
    "get_stock_chart": get_stock_chart
}
