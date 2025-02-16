from agents.agent import *
from crewai import Task

financial_report_task = Task(
    description=(
        """Retrieve and analyze financial data for {symbol} using the Yahoo Finance API. Extract key 
        metrics and provide a structured financial report, including:  
        - **Company Financials:** Revenue, net income, and key indicators.  
        - **Market Summary:** Company market standing and trends.  
        - **Stock Price Insights:** Latest stock data and historical trends.  
        If a query is provided, tailor the analysis accordingly while maintaining a comprehensive report."""
    ),
    agent=financial_report_agent,
    expected_output=(
        """A structured financial summary including:  
        - **Financial Metrics:** Revenue, profitability, and key indicators.  
        - **Market Performance:** Trends, risks, and industry standing.  
        - **Stock Price Data:** Latest price, historical trends, and insights.  
        If a query is provided, include tailored insights while ensuring a complete assessment."""
    )
)



investment_advisor_task = Task(
    description=(
        """Provide tailored investment recommendations for the given stock symbol {symbol}. 
        If a specific query is provided, incorporate it into your analysis. Otherwise, perform a comprehensive evaluation 
        by analyzing valuation metrics, historical performance, and current market trends. 
        Deliver actionable insights in a well-structured format to support informed investment decisions."""
    ),
    agent=investment_advisor_agent,
    expected_output=(
        """A structured analysis that includes tailored investment recommendations, 
        a detailed review of valuation metrics, historical performance, current market trends, 
        and, if applicable, responses to any specific query provided."""
    )
)

query_answerer_task_financial = Task(
    description=(
        """Answer specific queries related to financial data, market trends, and investment opportunities for the company {symbol}.
        Utilize your expertise to provide accurate and detailed responses to user questions."""
    ),
    agent=query_answerer_agent_financial,
    expected_output=(
        """Accurate and detailed responses to user queries related to financial data, market trends, and investment opportunities."""
    )
)

research_task = Task(
    description=("Search news about {topic}"),
    agent=news_researcher ,
    expected_output=(" A list of news articles about {topic} with the title, url, image and snippet")

)

reporting_task = Task(
    description=("Generate a report on the {topic}"),
    agent=reporting_analyst,
    expected_output=("""    A fully fledge reporting of the news articles.  Make sure you have the title, summary, url and the image.
    Formatted as markdown without '```
                     """)
)

medical_task = Task(
    description="Retrieve up to PubMed articles for {term}. Provide a short expert response if a query is given; otherwise, summarize key findings and recommendations.",
    agent=medical_agent,
    expected_output="A clear medical report: expert response for queries or a structured summary of key insights if no query is provided."
)