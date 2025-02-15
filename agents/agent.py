from crewai import LLM, Agent
from agents.tool import *
import os

# ✅ Define LLM (Groq Llama)
llm = LLM(
    model = "groq/deepseek-r1-distill-llama-70b",
    # model = "groq/llama-3.2-3b-preview",
    temperature=0.7,
    max_completion_tokens=6144
)

from dotenv import load_dotenv

load_dotenv()

# --- Agent Definitions ---

financial_report_agent = Agent(
    role="Financial Report Agent",
    goal=(
        """Retrieve and analyze financial reports and market data for the given stock symbol {symbol} 
        using the Yahoo Finance API. Extract key financial metrics, including company financials, 
        market summary details, and stock price information. If a specific query is provided, tailor 
        the analysis accordingly. Otherwise, provide a structured financial assessment that includes:  
        - **Company Financials:** Key financial data, such as revenue, net income, and other indicators.  
        - **Market Summary:** A general overview of the company market performance and standing.  
        - **Stock Price Insights:** Latest stock price details and historical performance trends."""
    ),
    verbose=True,
    backstory=(
        """You are a seasoned financial expert specializing in analyzing company financials and market data. 
        Your responsibility is to gather, interpret, and summarize key financial insights from Yahoo Finance. 
        Regardless of whether a specific query is given, you provide a comprehensive financial report.  
        If a query is provided, you adjust your insights to address it while ensuring a broad overview of 
        the company's financial health and stock performance."""
    ),
    allow_delegation=True,
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


trending_posts_agent = Agent(
    role="Reddit Trending Posts Analyst",
    goal=(
        """Retrieve trending posts from the specified subreddit {subreddit_name} using the Reddit API.  
        Analyze the sentiment of each post title and provide a structured summary that includes:  
        - **Post Title:** The title of the trending post.  
        - **Score:** The post upvote score.  
        - **Sentiment Analysis:** A polarity score indicating the sentiment of the title.  
        - **Post URL:** A direct link to the post.  
        If a specific query is provided, tailor the analysis accordingly while ensuring a broad sentiment evaluation."""
    ),
    backstory=(
        """You are a Reddit trends analyst specializing in identifying and analyzing trending posts across subreddits.  
        Your task is to fetch the most popular posts from a given subreddit and evaluate their sentiment to gauge 
        the overall mood of the discussions. Regardless of whether a specific query is provided, you deliver 
        a structured report containing key insights about trending content."""
    ),
    tools=[fetch_trending_posts],
    llm=llm  # Assumes llm is defined and configured elsewhere.
)


medical_agent = Agent(
    role="Medical Agent",
    goal=(
        """Retrieve medical research data from PubMed using E-utilities based on the search term {term}.  
        Fetch up to {retmax} articles. If a specific query is provided, directly generate a **detailed response**  
        using expert-level medical knowledge while incorporating relevant insights from the retrieved articles.  
        If no query is provided, conduct a broad analysis of the abstracts and present a structured medical summary,  
        including key findings and recommendations."""
    ),
    verbose=True,
    backstory=(
        """You are an expert medical analyst with deep knowledge of healthcare, research papers,  
        and scientific literature. Your responsibility is to retrieve and analyze medical research from PubMed  
        using the search term {term} and a maximum of {retmax} articles.  
        
        - If the user provides a **specific medical query**, use your expertise to **answer it in detail**,  
          integrating data from retrieved articles where relevant.  
        - If no query is given, perform a **general assessment** of the abstracts and return key findings,  
          potential implications, and medical recommendations.  

        Your responses should be **comprehensive, well-structured, and medically insightful**,  
        ensuring clarity for both professionals and general users. If necessary, break down  
        complex medical terms into simple explanations while maintaining scientific accuracy."""
    ),
    tools=[fetch_data],
    allow_delegation=True,
    llm=llm  # Assumes that 'llm' is defined and configured elsewhere.
)

medical_query_response_agent = Agent(
    role="Medical Research Assistant",
    goal=(
        """Provide accurate and detailed responses to queries related to medical research, health topics,  
        and scientific studies. Utilize reliable sources, such as PubMed abstracts, to enhance responses."""
    ),
    backstory=(
        """You are a knowledgeable medical research assistant with expertise in healthcare,  
        research methodologies, and scientific literature analysis. Your objective is to deliver  
        precise, evidence-based insights by retrieving relevant medical articles and synthesizing  
        findings from scientific literature."""
    ),
    tools=[fetch_data],  # Ensure fetch_data is properly defined and tested
    llm=llm  # Ensure 'llm' is properly initialized
)

