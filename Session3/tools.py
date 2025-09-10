"""
AI 에이전트가 사용할 도구들
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from langchain_core.tools import tool
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

@tool
def get_stock_basic_data(ticker: str) -> dict:
    """주식 기본 데이터를 가져옵니다"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        data = {
            "name": info.get('longName', ticker),
            "sector": info.get('sector', 'N/A'),
            "price": info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else 0),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "pb_ratio": info.get('priceToBook', 0),
            "dividend_yield": info.get('dividendYield', 0),
            "52w_high": info.get('fiftyTwoWeekHigh', 0),
            "52w_low": info.get('fiftyTwoWeekLow', 0),
        }
        return data
        
    except Exception as e:
        return {"error": str(e)}

@tool
def calculate_technical_indicators(ticker: str) -> dict:
    """기술지표를 계산합니다"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return {"error": "주가 데이터를 가져올 수 없습니다"}
        
        # 이동평균선 계산
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA60'] = hist['Close'].rolling(window=60).mean()
        
        # RSI 계산
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # ADX 계산
        high_low = hist['High'] - hist['Low']
        high_close = np.abs(hist['High'] - hist['Close'].shift())
        low_close = np.abs(hist['Low'] - hist['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        hist['ATR'] = tr.rolling(window=14).mean()
        
        plus_dm = hist['High'].diff()
        minus_dm = hist['Low'].diff() * -1
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_di = (plus_dm.rolling(window=14).mean() / hist['ATR']) * 100
        minus_di = (minus_dm.rolling(window=14).mean() / hist['ATR']) * 100
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        hist['ADX'] = dx.rolling(window=14).mean()
        
        latest_data = hist.iloc[-1]
        
        technical_data = {
            "current_price": latest_data['Close'],
            "ma20": latest_data['MA20'],
            "ma60": latest_data['MA60'],
            "rsi": latest_data['RSI'],
            "adx": latest_data['ADX'],
            "price_ma20_ratio": latest_data['Close'] / latest_data['MA20'],
            "ma20_ma60_trend": "상승" if latest_data['MA20'] > latest_data['MA60'] else "하락",
            "volume": latest_data['Volume'],
            "high_52w": hist['High'].max(),
            "low_52w": hist['Low'].min(),
            "hist_data": hist  # 차트용 데이터
        }
        
        return technical_data
        
    except Exception as e:
        return {"error": str(e)}

@tool
def analyze_trading_signals(technical_data: dict) -> dict:
    """매매 신호를 분석합니다"""
    if "error" in technical_data:
        return {"error": technical_data["error"]}
    
    ma20_above_ma60 = technical_data["ma20"] > technical_data["ma60"]
    price_ma20_ratio = technical_data["price_ma20_ratio"]
    rsi = technical_data["rsi"]
    adx = technical_data["adx"]
    
    long_conditions = {
        "trend": ma20_above_ma60,
        "position": price_ma20_ratio < 1.02,
        "momentum": rsi < 70,
        "strength": adx > 25
    }
    
    short_conditions = {
        "trend": not ma20_above_ma60,
        "position": price_ma20_ratio > 0.98,
        "momentum": rsi > 30,
        "strength": adx > 25
    }
    
    long_exit_conditions = {
        "trend_break": not ma20_above_ma60,
        "overbought": rsi > 75,
        "weak_trend": adx < 20,
        "distance": price_ma20_ratio > 1.08
    }
    
    short_exit_conditions = {
        "trend_break": ma20_above_ma60,
        "oversold": rsi < 25,
        "weak_trend": adx < 20,
        "distance": price_ma20_ratio < 0.92
    }
    
    long_score = sum(long_conditions.values())
    short_score = sum(short_conditions.values())
    long_exit_score = sum(long_exit_conditions.values())
    short_exit_score = sum(short_exit_conditions.values())
    
    if long_score == 4:
        entry_signal = "강력 매수"
    elif long_score == 3:
        entry_signal = "매수"
    elif short_score == 4:
        entry_signal = "강력 매도"
    elif short_score == 3:
        entry_signal = "매도"
    else:
        entry_signal = "중립"
    
    return {
        "entry_signal": entry_signal,
        "long_score": long_score,
        "short_score": short_score,
        "long_exit_score": long_exit_score,
        "short_exit_score": short_exit_score,
        "long_conditions": long_conditions,
        "short_conditions": short_conditions,
        "long_exit_conditions": long_exit_conditions,
        "short_exit_conditions": short_exit_conditions
    }

@tool
def search_company_news_tavily(ticker: str, company_name: str) -> dict:
    """Tavily를 사용해 회사 관련 뉴스를 검색합니다"""
    try:
        client = TavilyClient()
        
        search_queries = [
            f"{ticker} {company_name} news 2024 2025",
            f"{company_name} earnings stock price",
            f"{ticker} analyst rating upgrade downgrade"
        ]
        
        all_news = []
        for query in search_queries:
            try:
                search_results = client.search(
                    query,
                    max_results=3,
                    include_raw_content=True,
                    search_depth="advanced"
                )
                if 'results' in search_results:
                    all_news.extend(search_results['results'])
            except:
                continue
        
        unique_news = []
        seen_urls = set()
        
        for news in all_news:
            url = news.get('url', '')
            if url not in seen_urls and len(unique_news) < 6:
                seen_urls.add(url)
                unique_news.append({
                    'title': news.get('title', ''),
                    'content': news.get('content', ''),
                    'url': news.get('url', ''),
                    'score': news.get('score', 0)
                })
        
        return {
            "success": True,
            "news_count": len(unique_news),
            "news_articles": unique_news,
            "search_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def analyze_news_sentiment_ai(news_data: dict, ticker: str, company_name: str) -> dict:
    """수집된 뉴스의 감정을 분석합니다"""
    if not news_data.get("success") or not news_data.get("news_articles"):
        return {"sentiment": "중립", "score": 0, "analysis": "뉴스 데이터 없음"}
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    news_summary = ""
    for i, article in enumerate(news_data["news_articles"][:5]):
        news_summary += f"\n{i+1}. {article['title']}\n{article['content'][:300]}...\n"
    
    prompt = ChatPromptTemplate.from_template("""
    당신은 금융 뉴스 감정 분석 전문가입니다.
    
    {ticker} ({company_name}) 관련 최신 뉴스들을 분석하여 투자 관점에서 감정을 평가해주세요.
    
    뉴스 내용:
    {news_summary}
    
    **감정 점수 (-10 ~ +10):**
    - 매우 부정적: -8~-10 (주가 하락 요인)
    - 부정적: -4~-7 (약한 하락 요인)
    - 중립적: -3~+3 (큰 영향 없음)
    - 긍정적: +4~+7 (약한 상승 요인)
    - 매우 긍정적: +8~+10 (주가 상승 요인)
    
    **반드시 다음 형식으로 답변하세요:**
    
    ## 뉴스 감정 분석 결과
    
    **감정 점수**: [점수]/10 → [매우 부정적/부정적/중립적/긍정적/매우 긍정적]
    
    **긍정적 뉴스** (상승 요인):
    - [구체적 내용 1]
    - [구체적 내용 2]
    
    **부정적 뉴스** (하락 요인):
    - [구체적 내용 1]
    - [구체적 내용 2]
    
    **주요 이벤트**:
    - [중요한 발표나 이벤트]
    
    **투자 임팩트**:
    [뉴스가 주가에 미칠 영향 예측]
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        analysis = chain.invoke({
            "ticker": ticker,
            "company_name": company_name,
            "news_summary": news_summary
        })
        
        score = 0
        if "매우 긍정적" in analysis:
            score = 8
        elif "긍정적" in analysis:
            score = 5
        elif "매우 부정적" in analysis:
            score = -8
        elif "부정적" in analysis:
            score = -5
        else:
            score = 0
            
        return {
            "sentiment": "긍정적" if score > 3 else "부정적" if score < -3 else "중립적",
            "score": score,
            "analysis": analysis
        }
        
    except Exception as e:
        return {
            "sentiment": "중립적",
            "score": 0,
            "analysis": f"분석 실패: {str(e)}"
        }

@tool
def create_technical_chart(ticker: str, hist_data: pd.DataFrame) -> str:
    """주가 + MA + RSI 차트를 생성합니다"""
    try:
        # 최근 6개월 데이터만 사용
        data_6m = hist_data.tail(120)  # 약 6개월
        
        # 한글 폰트 문제 해결을 위해 영어만 사용
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
        
        # 상단: 주가 + MA
        ax1.plot(data_6m.index, data_6m['Close'], label='Price', linewidth=1.5)
        ax1.plot(data_6m.index, data_6m['MA20'], label='MA20', linewidth=1, alpha=0.7)
        ax1.plot(data_6m.index, data_6m['MA60'], label='MA60', linewidth=1, alpha=0.7)
        
        ax1.set_title(f'{ticker} - Price and Moving Averages (6 Months)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 하단: RSI
        ax2.plot(data_6m.index, data_6m['RSI'], label='RSI', linewidth=2, color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.fill_between(data_6m.index, 30, 70, alpha=0.1, color='gray')
        
        ax2.set_title(f'{ticker} - RSI (14)', fontsize=12)
        ax2.set_ylabel('RSI', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 날짜 포맷팅
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        # 파일 저장
        filename = f"{ticker}_technical_chart.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
        
    except Exception as e:
        return f"차트 생성 실패: {str(e)}"