from datetime import datetime
import pytz
import yfinance as yf
from datetime import datetime


def get_current_time(timezone: str = 'Asia/Seoul'):
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    now_timezone = f'{now} {timezone}'
    print(now_timezone)
    return now_timezone

def get_yf_stock_info(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    print(info)
    return str(info)

def get_yf_stock_history(ticker: str, period: str):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    history_md = history.to_markdown()
    print(history_md)
    return history_md

def get_yf_stock_recommendations(ticker: str):
    stock = yf.Ticker(ticker)
    recommendations = stock.recommendations
    recommendations_md = recommendations.to_markdown()
    print(recommendations_md)
    return recommendations_md


tools = [
    {
        "type":"function",
        "function": {
            "name": "get_current_time",
            "description": "해당 타임존의 현재 날짜와 시간을 타임존과 함께 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    'timezone': {
                        'type': 'string',
                        'description': '현재 날짜와 시간을 반환받고자 하는 타임존을 입력하세요. (예: Asian/Seoul)',
                    },
                },
                "required": ['timezone'],
            },
        }
    },

    {
        "type":"function",
        "function": {
            "name": "get_yf_stock_info",
            "description": "해당 종목의 Yahoo Finance 제공 정보를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    'ticker': {
                        'type': 'string',
                        'description': 'Yahoo Finance 에서 정보를 조회할 종목의 티커를 입력하세요. (예: TSLA)',
                    },
                },
                "required": ['ticker'],
            },
        }
    },
    {
        "type":"function",
        "function": {
            "name": "get_yf_stock_history",
            "description": "해당 종목의 Yahoo Finance 제공 주가 정보를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    'ticker': {
                        'type': 'string',
                        'description': 'Yahoo Finance 에서 주가 정보를 조회할 종목의 티커를 입력하세요. (예: TSLA)',
                    },
                    'period': {
                        'type': 'string',
                        'description': '주가 정보를 알고 싶은 기간을 입력하세요. (예: 1d, 5d, 1mo, 1y, 5y)',
                    },
                },
                "required": ['ticker', 'period'],
            },
        }
    },
    {
        "type":"function",
        "function": {
            "name": "get_yf_stock_recommendations",
            "description": "해당 종목에 대한 Yahoo Finance 제공 추천 정보를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    'ticker': {
                        'type': 'string',
                        'description': 'Yahoo Finance 에서 추천 정보를 조회할 종목의 티커를 입력하세요. (예: TSLA)',
                    },
                },
                "required": ['ticker'],
            },
        }
    },      

]

if __name__ == '__main__':
    get_current_time('Europe/London')
    get_yf_stock_info('NVDA')
    get_yf_stock_history('TSLA', '10d')
    get_yf_stock_recommendations('TSLA')