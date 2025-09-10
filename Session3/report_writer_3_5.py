#!/usr/bin/env python3
"""
간단한 투자 리서치 AI 봇 v3.5
- tools.py 분리
- 상세한 보고서 작성 (각 의견 300자 이상)
- 기술 차트 이미지 포함

필요 패키지:
pip install langgraph langchain-openai yfinance python-dotenv pandas numpy tavily-python matplotlib
"""

import os
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# tools.py에서 도구들 import
from tools import (
    get_stock_basic_data,
    calculate_technical_indicators, 
    analyze_trading_signals,
    search_company_news_tavily,
    analyze_news_sentiment_ai,
    create_technical_chart
)

load_dotenv()

if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    print("오류: .env 파일에 OPENAI_API_KEY와 TAVILY_API_KEY를 설정해주세요!")
    exit(1)

# =============================================================================
# 상태 정의
# =============================================================================

class InvestmentState(TypedDict):
    ticker: str
    stock_data: dict
    technical_data: dict
    news_data: dict
    chart_filename: str
    fundamental_analysis: str
    technical_analysis: str
    news_analysis: str
    draft_report: str
    final_report: str
    current_step: str

# =============================================================================
# AI 에이전트들 (LangGraph 노드 함수들)
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

def fundamental_analyst(state: InvestmentState) -> InvestmentState:
    """기본 분석 에이전트"""
    print("\n[기본분석가 작업 중...]")
    
    ticker = state["ticker"]
    data = get_stock_basic_data.invoke({"ticker": ticker})
    state["stock_data"] = data
    
    if "error" in data:
        state["fundamental_analysis"] = f"데이터 오류: {data['error']}"
        return state
    
    prompt = f"""
당신은 경험 많은 기본분석 전문가입니다.

{ticker} ({data['name']}) 기업의 기본 분석을 상세히 해주세요.

데이터:
- 현재 주가: ${data['price']:.2f}
- 시가총액: ${data['market_cap']:,}
- P/E 비율: {data['pe_ratio']:.2f}
- P/B 비율: {data['pb_ratio']:.2f}
- 배당수익률: {data['dividend_yield']:.2%}
- 52주 최고가: ${data['52w_high']:.2f}
- 52주 최저가: ${data['52w_low']:.2f}
- 섹터: {data['sector']}

반드시 300자 이상의 상세한 분석을 제공하세요:

1. 밸류에이션 분석: P/E, P/B 비율의 적정성을 동종업계 평균과 비교하여 구체적으로 분석
2. 재무 건전성: 현재 재무 상태의 강점과 약점을 구체적 수치를 들어 설명
3. 배당 정책: 배당수익률과 배당 정책의 지속가능성 평가
4. 섹터 분석: {data['sector']} 섹터 내에서의 경쟁적 위치와 성장 전망
5. 투자 매력도: 현재 주가 수준에서의 투자 가치와 위험 요인

각 항목마다 구체적인 근거와 수치를 제시하고, 최종적으로 기본분석 관점에서의 명확한 투자 의견을 제시해주세요.
"""

    messages = [SystemMessage(content="당신은 기업 기본분석 전문가입니다."), 
                HumanMessage(content=prompt)]
    
    response = llm.invoke(messages)
    state["fundamental_analysis"] = response.content
    state["current_step"] = "fundamental_done"
    
    print("기본분석 완료!")
    return state

def technical_analyst(state: InvestmentState) -> InvestmentState:
    """기술 분석 에이전트"""
    print("[기술분석가 작업 중...]")
    
    ticker = state["ticker"]
    
    # 기술지표 계산
    tech_data = calculate_technical_indicators.invoke({"ticker": ticker})
    state["technical_data"] = tech_data
    
    if "error" in tech_data:
        state["technical_analysis"] = f"기술지표 계산 실패: {tech_data['error']}"
        return state
    
    # 매매신호 분석
    signals = analyze_trading_signals.invoke({"technical_data": tech_data})
    
    # 차트 생성
    if "hist_data" in tech_data:
        chart_file = create_technical_chart.invoke({
            "ticker": ticker, 
            "hist_data": tech_data["hist_data"]
        })
        state["chart_filename"] = chart_file
    
    prompt = f"""
당신은 경험 많은 기술분석 전문가입니다.

{ticker} 종목의 상세 기술적 분석을 해주세요.

기술지표 현황:
- 현재 주가: ${tech_data['current_price']:.2f}
- MA20: ${tech_data['ma20']:.2f}
- MA60: ${tech_data['ma60']:.2f}
- Price/MA20 비율: {tech_data['price_ma20_ratio']:.3f}
- RSI(14): {tech_data['rsi']:.1f}
- ADX: {tech_data['adx']:.1f}

매매신호: {signals['entry_signal']} (롱 {signals['long_score']}/4, 숏 {signals['short_score']}/4)

반드시 300자 이상의 상세한 분석을 제공하세요:

1. 추세 분석: MA20({tech_data['ma20']:.2f}) vs MA60({tech_data['ma60']:.2f}) 관계를 통한 중장기 추세 방향성 분석
2. 현재 위치: 주가가 MA20 대비 {((tech_data['price_ma20_ratio']-1)*100):+.1f}% 위치에 있는 의미와 과열/과매도 여부
3. 모멘텀 분석: RSI {tech_data['rsi']:.1f}의 의미와 향후 반전 가능성, 다이버전스 확인
4. 추세 강도: ADX {tech_data['adx']:.1f}를 통한 현재 추세의 지속 가능성 평가
5. 매매 전략: 진입/청산 시점, 목표가 ${tech_data['current_price']*1.08:.2f}, 손절가 ${tech_data['current_price']*0.95:.2f} 설정 근거

각 지표의 상호작용과 시장 상황을 종합적으로 고려하여 구체적인 매매 전략을 제시해주세요.
"""

    messages = [SystemMessage(content="당신은 기술분석 전문가입니다."), 
                HumanMessage(content=prompt)]
    
    response = llm.invoke(messages)
    state["technical_analysis"] = response.content
    state["current_step"] = "technical_done"
    
    print("기술분석 완료!")
    return state

def news_analyst(state: InvestmentState) -> InvestmentState:
    """뉴스 분석 에이전트"""
    print("[뉴스분석가 작업 중...]")
    
    ticker = state["ticker"]
    data = state["stock_data"]
    
    # 뉴스 검색
    news_data = search_company_news_tavily.invoke({
        "ticker": ticker, 
        "company_name": data.get('name', ticker)
    })
    state["news_data"] = news_data
    
    if not news_data.get("success"):
        state["news_analysis"] = f"뉴스 검색 실패: {news_data.get('error', 'Unknown error')}"
        return state
    
    # 감정 분석
    sentiment_result = analyze_news_sentiment_ai.invoke({
        "news_data": news_data,
        "ticker": ticker,
        "company_name": data.get('name', ticker)
    })
    
    prompt = f"""
당신은 시장 뉴스 및 정서 분석 전문가입니다.

{ticker} ({data.get('name', ticker)}) 종목의 뉴스 분석을 상세히 해주세요.

뉴스 검색 결과:
- 검색된 뉴스: {news_data.get('news_count', 0)}개
- 감정 점수: {sentiment_result['score']}/10 ({sentiment_result['sentiment']})

AI 감정 분석:
{sentiment_result['analysis']}

반드시 300자 이상의 상세한 분석을 제공하세요:

1. 시장 정서 평가: 현재 {sentiment_result['sentiment']} 정서(점수: {sentiment_result['score']}/10)의 배경과 근거
2. 주요 호재 분석: 검색된 뉴스 중 주가 상승 요인이 될 수 있는 구체적 내용들
3. 주요 악재 분석: 투자 리스크가 될 수 있는 부정적 뉴스들의 임팩트 평가  
4. 섹터 동향: {data.get('sector', 'N/A')} 섹터 전반의 시장 환경과 트렌드 분석
5. 향후 모니터링: 주가에 중대한 영향을 미칠 수 있는 예정된 이벤트나 발표 일정

각 뉴스의 시장 임팩트를 단기(1-3개월), 중장기(6-12개월) 관점에서 구분하여 분석해주세요.
"""

    messages = [SystemMessage(content="당신은 시장 뉴스 및 정서 분석 전문가입니다."), 
                HumanMessage(content=prompt)]
    
    response = llm.invoke(messages)
    state["news_analysis"] = response.content
    state["current_step"] = "news_done"
    
    print("뉴스분석 완료!")
    return state

def report_writer(state: InvestmentState) -> InvestmentState:
    """보고서 작성 에이전트"""
    print("[보고서작성가 작업 중...]")
    
    ticker = state["ticker"]
    data = state["stock_data"]
    chart_file = state.get("chart_filename", "")
    
    prompt = f"""
{ticker} ({data['name']}) 종목에 대한 종합 투자 보고서를 작성해주세요.

분석 결과들:

## 기본분석:
{state['fundamental_analysis']}

## 기술분석:
{state['technical_analysis']}

## 뉴스분석:
{state['news_analysis']}

다음 형식으로 상세한 보고서를 작성하세요:

# {ticker} 투자 분석 보고서 v3.5

## 투자 요약
- **투자의견**: [매수/보유/매도]
- **목표주가**: $[금액]
- **투자기간**: [단기/중기/장기]
- **신뢰도**: [높음/보통/낮음]

## 기업 개요
- 회사명: {data['name']}
- 섹터: {data['sector']}
- 현재주가: ${data['price']:.2f}
- 시가총액: ${data['market_cap']:,}

## 기술적 분석 차트
![Technical Chart]({chart_file})

## 분석가별 상세 의견

### 기본분석가 의견
{state['fundamental_analysis'][:500]}...

### 기술분석가 의견  
{state['technical_analysis'][:500]}...

### 뉴스분석가 의견
{state['news_analysis'][:500]}...

## 투자 리스크
- **높은 리스크**: [최우선 리스크]
- **중간 리스크**: [2차 리스크]
- **모니터링 포인트**: [관찰 사항]

## 최종 결론 및 실행 전략
[3개 분석 종합한 최종 판단과 실행 방안]

각 섹션을 상세히 작성해주세요.
"""

    messages = [SystemMessage(content="당신은 투자 보고서 작성 전문가입니다."), 
                HumanMessage(content=prompt)]
    
    response = llm.invoke(messages)
    state["draft_report"] = response.content
    state["current_step"] = "report_done"
    
    print("보고서 작성 완료!")
    return state

def supervisor(state: InvestmentState) -> InvestmentState:
    """감독 에이전트"""
    print("[감독관 검토 중...]")
    
    ticker = state["ticker"]
    
    prompt = f"""
{ticker} 종목 투자 보고서를 최종 검토해주세요.

초안 보고서:
{state['draft_report']}

검토 기준:
1. 기본분석 밸류에이션 의견이 구체적으로 반영되었는가?
2. 기술분석 매매신호와 차트가 명확히 포함되었는가?
3. 뉴스분석 감정점수와 실제 뉴스가 반영되었는가?
4. 각 분석가 의견이 300자 이상으로 상세한가?

필요시 보완하여 최종 보고서를 완성하세요. 완성된 보고서만 출력하세요.
"""

    messages = [SystemMessage(content="당신은 투자 보고서 품질 관리 전문가입니다."), 
                HumanMessage(content=prompt)]
    
    response = llm.invoke(messages)
    state["final_report"] = response.content
    state["current_step"] = "completed"
    
    print("감독 검토 완료!")
    return state

# =============================================================================
# 워크플로우 및 실행
# =============================================================================

def create_workflow():
    workflow = StateGraph(InvestmentState)
    
    workflow.add_node("fundamental", fundamental_analyst)
    workflow.add_node("technical", technical_analyst)
    workflow.add_node("news", news_analyst)
    workflow.add_node("report", report_writer)
    workflow.add_node("supervisor", supervisor)
    
    workflow.set_entry_point("fundamental")
    workflow.add_edge("fundamental", "technical")
    workflow.add_edge("technical", "news")
    workflow.add_edge("news", "report")
    workflow.add_edge("report", "supervisor")
    workflow.add_edge("supervisor", END)

    app = workflow.compile()

    absolute_path = os.path.abspath(__file__)
    app.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py', '.png'))

    return app

def save_report(ticker: str, report: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_investment_report_v35_{timestamp}.md"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"보고서 저장: {filename}")
        return filename
    except Exception as e:
        print(f"보고서 저장 실패: {e}")
        return None

def analyze_stock(ticker: str):
    print(f"\n{ticker} [분석 시작...]")
    
    initial_state = {
        "ticker": ticker.upper(),
        "stock_data": {},
        "technical_data": {},
        "news_data": {},
        "chart_filename": "",
        "fundamental_analysis": "",
        "technical_analysis": "",
        "news_analysis": "",
        "draft_report": "",
        "final_report": "",
        "current_step": "started"
    }
    
    try:
        app = create_workflow() # 랭그래프 워크플로우 객체 생성
        final_state = app.invoke(initial_state) # 작업 실행
        
        print(f"\n" + "="*60)
        print(f"{ticker} 투자 분석 보고서 v3.5")
        print("="*60)
        print(final_state["final_report"])
        print("="*60)
        
        # 차트 파일 확인
        chart_file = final_state.get("chart_filename", "")
        if chart_file and os.path.exists(chart_file):
            print(f"차트 파일 생성: {chart_file}")
        
        filename = save_report(ticker, final_state["final_report"])
        return final_state, filename
        
    except Exception as e:
        print(f"분석 오류: {e}")
        return None, None

def main():
    print("AI 투자 리서치 봇 v3.5")
    print("=" * 50)
    print("업데이트: tools.py 분리 + 상세보고서 + 기술차트")
    print("'exit' 입력으로 종료")
    print("=" * 50)
    
    while True:
        ticker = input("\n안녕하세요? 분석할 종목 코드를 입력하세요. (예: AAPL): ").strip().upper()
        
        if ticker == 'EXIT':
            print("종료합니다!")
            break
        
        if not ticker:
            print("종목 코드를 입력해주세요.")
            continue
        
        try:
            result, filename = analyze_stock(ticker)
            
            if result and filename:
                print(f"\n{ticker} 분석 완료!")
                print(f"보고서: {filename}")
                
                chart_file = result.get("chart_filename", "")
                if chart_file:
                    print(f"차트: {chart_file}")
            else:
                print(f"{ticker} 분석 실패")
                
        except KeyboardInterrupt:
            print("\n분석 중단됨")
            continue
        except Exception as e:
            print(f"오류: {e}")
            continue

if __name__ == "__main__":
    main()