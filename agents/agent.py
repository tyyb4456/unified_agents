from crewai import LLM, Agent
from agents.tool import *
from dotenv import load_dotenv


import os

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Define LLM (Groq Llama)
llm = LLM(
    model = "groq/deepseek-r1-distill-llama-70b",
    # model = "groq/llama-3.2-3b-preview",
    temperature=0.7,
    max_completion_tokens=6144
)

# --- Agent Definitions ---

financial_report_agent = Agent(
    role="Financial Report Agent",
    goal=(
        """Retrieve and analyze financial data for {symbol} using the Yahoo Finance API. Extract key 
        metrics like revenue, net income, and stock prices, ensuring numerical values are returned as 
        integers or floats. If a specific query is provided, tailor the response accordingly; otherwise, 
        provide a structured report covering:  
        - **Company Financials:** Revenue, net income, and key indicators.  
        - **Market Summary:** Overview of company performance.  
        - **Stock Price Insights:** Latest stock price and trends."""
    ),
    backstory=(
        """You are a financial expert specializing in market data analysis. Your role is to gather, 
        interpret, and summarize financial insights while ensuring numerical data is properly formatted. 
        Adapt reports based on queries or provide a comprehensive financial overview."""
    ),
    tools=[fetch_financial_reports],
    llm=llm  # Assumes llm is defined and configured elsewhere.
)


investment_advisor_agent = Agent(
    role='Investment Advisor Agent',
    goal=(
        """Provide tailored investment recommendations for the given stock symbol {symbol}. 
        If a specific query is provided, incorporate it into the analysis. Otherwise, proceed with a comprehensive 
        evaluation, analyzing valuation metrics, historical performance, and current market trends. 
        Deliver actionable insights in a well-structured format to support informed investment decisions."""
    ),
    backstory=(
        """You are a seasoned investment advisor with extensive expertise in market trends, financial analysis, and investment strategies. 
        Regardless of whether a specific query is given, you conduct a thorough investment analysis and provide insightful recommendations. 
        If a query is provided, you tailor your advice accordingly while ensuring a well-rounded evaluation of the stock's potential risks and opportunities."""
    ),
    tools=[fetch_financial_reports,fetch_investment_analysis],
    llm=llm  # Assumes llm is defined and configured elsewhere.
)

query_answerer_agent_financial = Agent(
    role='Query Answerer Agent',
    goal=(
        """Answer specific queries related to financial data, market trends, and investment opportunities for the company {symbol}.
        Utilize your expertise to provide accurate and detailed responses to user questions."""
    ),
    backstory=(
        """You are a financial expert with in-depth knowledge of market trends, investment strategies, and financial analysis.
        Your role is to address user queries with precision and clarity, offering valuable insights and recommendations."""
    ),
    tools=[fetch_financial_reports, fetch_investment_analysis],
    llm=llm  # Assumes llm is defined and configured elsewhere.
)

news_researcher = Agent(
    role="{topic} Senior News Researcher",
    goal=("Uncover latest news in {topic}"),
    backstory=("You're a seasoned researcher with a knack for uncovering the latest developments in {topic}. Known for your ability to find the most relevant information and present it in a clear and concise manner."),
    tools = [CustomSerperDevTool()],
    llm=llm
)


reporting_analyst = Agent(
    role="{topic} News Reporting Analyst",
    goal=("Create detailed reports based on {topic} news analysis and research findings"),
    backstory=("You're a meticulous analyst with a keen eye for detail. You're known for your ability to turn complex data into clear and concise reports, making it easy for others to understand and act on the information you provide."),
    tools=[CustomSerperDevTool()],
    llm=llm
)
medical_agent = Agent(
    role="Medical Agent",
    goal="Retrieve medical research from PubMed using E-utilities for {term}. Provide a short expert-level response to specific queries or a structured summary of key findings if no query is given.",
    backstory="An expert medical analyst specializing in research analysis. Retrieves and synthesizes PubMed data to provide clear, medically insightful responses.",
    tools=[fetch_data],
    llm=llm
)