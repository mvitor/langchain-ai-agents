import os
import time
import schedule
import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime, timedelta
from typing import Dict, List, TypedDict, Optional
import logging
import json
from pathlib import Path

from langgraph.prebuilt import create_react_agent

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain.agents.agent_toolkits.gmail.toolkit import GmailToolkit
from langchain_google_community.gmail.utils  import build_resource_service, get_gmail_credentials

from google.oauth2.credentials import Credentials

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bitcoin_agent.log"), logging.StreamHandler()],
)
logger = logging.getLogger("bitcoin_agent")

# Create directory for charts if it doesn't exist
CHARTS_DIR = Path("bitcoin_charts")
CHARTS_DIR.mkdir(exist_ok=True)

# Create directory for analysis history
HISTORY_DIR = Path("bitcoin_analysis_history")
HISTORY_DIR.mkdir(exist_ok=True)


# Define the state schema as a TypedDict
class AgentState(TypedDict):
    messages: List
    current_price: Dict
    historical_prices: Dict
    analysis: Dict
    error: str
    timestamp: str
    tool_calls: Optional[List[Dict]] = None

def get_gmail_credentials(token_file, scopes, client_secrets_file):
    """Get or refresh Gmail API credentials."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import os

    creds = None
    # Check if token file exists
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_info(
            json.loads(open(token_file).read()), scopes
        )

    # If no valid credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes
            )
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
        
        return creds

def create_gmail_toolkit():
    # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
    # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    return GmailToolkit(api_resource=api_resource)

# Define functions for the nodes
def get_current_bitcoin_price(state: AgentState) -> AgentState:
    """Get current Bitcoin price from CoinGecko API."""
    try:
        logger.info("Fetching current Bitcoin price...")
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": "bitcoin",
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            },
        )
        response.raise_for_status()

        # Create a new state with updated current_price
        new_state = state.copy()
        new_state["current_price"] = response.json()["bitcoin"]
        logger.info(f"Current BTC price: ${new_state['current_price']['usd']}")
        return new_state
    except Exception as e:
        logger.error(f"Error getting current price: {str(e)}")
        # Create a new state with error
        new_state = state.copy()
        new_state["error"] = f"Error getting current price: {str(e)}"
        return new_state


def get_historical_bitcoin_prices(state: AgentState) -> AgentState:
    """Get historical Bitcoin prices for the last 30 days."""
    try:
        logger.info("Fetching historical Bitcoin prices for the last 30 days...")
        # Calculate dates for 30 days ago
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Format dates for API
        end_timestamp = int(end_date.timestamp())
        start_timestamp = int(start_date.timestamp())

        response = requests.get(
            f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range",
            params={"vs_currency": "usd", "from": start_timestamp, "to": end_timestamp},
        )
        response.raise_for_status()

        # Create a new state with updated historical_prices
        new_state = state.copy()
        new_state["historical_prices"] = response.json()
        logger.info(
            f"Retrieved {len(new_state['historical_prices']['prices'])} historical price points"
        )
        return new_state
    except Exception as e:
        logger.error(f"Error getting historical prices: {str(e)}")
        # Create a new state with error
        new_state = state.copy()
        new_state["error"] = f"Error getting historical prices: {str(e)}"
        return new_state


def generate_analysis(state: AgentState) -> AgentState:
    """Generate analysis using Llama 3.1 8B."""
    try:
        if state.get("error"):
            return state

        logger.info("Generating Bitcoin price analysis...")
        # Create a DataFrame from historical price data
        price_data = pd.DataFrame(
            state["historical_prices"]["prices"], columns=["timestamp", "price"]
        )
        price_data["date"] = pd.to_datetime(price_data["timestamp"], unit="ms")

        # Prepare data for the LLM
        current_price = state["current_price"]["usd"]
        price_change_24h = state["current_price"]["usd_24h_change"]

        # Calculate some basic metrics
        latest_prices = price_data.tail(7)["price"].tolist()
        oldest_prices = price_data.head(7)["price"].tolist()
        avg_latest = sum(latest_prices) / len(latest_prices)
        avg_oldest = sum(oldest_prices) / len(oldest_prices)
        monthly_trend = ((avg_latest / avg_oldest) - 1) * 100

        # Generate chart
        plt.figure(figsize=(10, 6))
        plt.plot(price_data["date"], price_data["price"])
        plt.title("Bitcoin Price - Last 30 Days")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True)
        plt.tight_layout()

        # Get current timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save chart to disk
        chart_filename = CHARTS_DIR / f"bitcoin_price_chart_{timestamp}.png"
        plt.savefig(chart_filename)
        logger.info(f"Saved chart to {chart_filename}")

        # Also save chart to BytesIO object for embedding in the output
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.read()).decode("utf-8")
        position_value = 2009
        wbtc_value = 0
        usdt_value = 2011
        min_value = 87313
        max_value = 93879.0
        min_value_adjusted = (min_value)-100
        max_value_adjusted = (max_value)+100
        pool_data = {
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "monthly_trend": monthly_trend,
            "position_value": position_value,
            "usdt_value": usdt_value,
            "wbtc_value": wbtc_value,
            "min_value": min_value,
            "max_value": max_value,
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful financial assistant specializing in DeFi Pools",
                ),
                (
                    "human",
                    """
Analyze the provided Bitcoin (BTC) liquidity pool data on Uniswap v4 (Polygon network) and suggest specific adjustments to maximize returns with the available liquidity. Base your recommendations on the market trends and pool parameters below, considering price volatility, fee earnings, liquidity distribution, and Polygon’s low gas fees. Provide clear, actionable suggestions (e.g., adjust USDT/WBTC allocation, modify price range) with brief reasoning, prioritizing strategies that balance risk and reward. Focus on whether now is a good time to rearrange the pool, with specific suggestions for the lower and higher price bounds, and consider other optimizations like token allocation. All recommendations must use the current liquidity (no additional funds). Conclude with one primary recommendation and a disclaimer.

**Market Data:**
- Current Bitcoin Price: ${current_price:.2f}
- 24-Hour Price Change: {price_change_24h:.2f}%
- 30-Day Price Trend: {monthly_trend:.2f}%

**Uniswap v4 Pool Details:**
- Position Value: ${position_value:.2f}
- USDT Allocation: ${usdt_value:.2f}
- WBTC Allocation: {wbtc_value:.6f} WBTC
- Lower Price Bound: ${min_value:.2f}
- Higher Price Bound: ${max_value:.2f}

**Instructions:**
1. Analyze the 24-hour ({price_change_24h:.2f}%) and 30-day ({monthly_trend:.2f}%) trends to determine Bitcoin’s price direction (bullish, bearish, or stable).
2. Evaluate the pool’s position value (${position_value:.2f}), USDT/WBTC allocation (${usdt_value:.2f}/{wbtc_value:.6f} WBTC), and price range (${min_value:.2f}–${max_value:.2f}) relative to the current price (${current_price:.2f}).
3. Suggest 2–3 adjustments to optimize returns, such as:
   - Rebalancing USDT/WBTC allocation (e.g., increase USDT for stability in a volatile market).
   - Adjusting the price range (e.g., widen for more fees, narrow for concentrated liquidity).
   - Modifying position size within the current liquidity.
4. Provide reasoning tied to the data (e.g., “A {monthly_trend:.2f}% upward trend suggests a wider range to capture fees”).
5. Assess if now is a good time to rearrange the pool, focusing on price range adjustments.
6. Conclude with one primary recommendation (e.g., “Widen the price range to ${min_value_adjusted:.2f}–${max_value_adjusted:.2f}”).

**Output Format:**
- **Analysis**: Summarize price trends and pool status (1–2 sentences).
- **Suggestions**: List 2–3 adjustments with reasoning.
- **Primary Recommendation**: One clear action, including specific lower and higher price bounds.
- **Disclaimer**: “These suggestions are based on data analysis and are not professional financial advice. Consult a financial advisor.”

**Constraints:**
- Work within the current position value (${position_value:.2f}).
- Ensure feasibility on Uniswap v4 (Polygon network).
- Leverage Polygon’s low gas fees for adjustments.

** Suggest a Lower and higher range if any change is required. Use the following format to provide the suggestions

Lower Price: Lowerprice
Higher Price: HighPrice


I need your analysis on whether this is a good time to rearrange my pool, with specific suggestions for the lower and higher price bounds, alongside other optimization strategies.
Send an e-mail to marciovitormatos@gmail.com with every analysis
            """,
                ),
            ]
        )
        # Call the LLM
        logger.info("Calling Llama3.1:8b model for analysis...")
        toolkit = create_gmail_toolkit()
        #llm = Ollama(model="llama3.1:8b")
        llm = ChatOllama(
            model="llama3.1",
            temperature=0,
        )
        #.bind_tools([toolkit])        
        #tools = toolkit.get_tools()
        #llm = create_react_agent(llm, tools)


        # chain = prompt.format(
        #         current_price=pool_data["current_price"],
        #         price_change_24h=pool_data["price_change_24h"],
        #         monthly_trend=pool_data["monthly_trend"],
        #         position_value=pool_data["position_value"],
        #         usdt_value=pool_data["usdt_value"],
        #         wbtc_value=pool_data["wbtc_value"],
        #         min_value=pool_data["min_value"],
        #         max_value=pool_data["max_value"],
        #         min_value_adjusted=min_value_adjusted,
        #         max_value_adjusted=max_value_adjusted
        chain = prompt | llm | StrOutputParser()
        analysis = chain.invoke(
            input={
                "current_price": pool_data["current_price"],
                "price_change_24h": pool_data["price_change_24h"],
                "monthly_trend": pool_data["monthly_trend"],
                "position_value": pool_data["position_value"],
                "usdt_value": pool_data["usdt_value"],
                "wbtc_value": pool_data["wbtc_value"],
                "min_value": pool_data["min_value"],
                "max_value": pool_data["max_value"],
                "min_value_adjusted": min_value_adjusted,
                "max_value_adjusted": max_value_adjusted,
            }
        )
        print(analysis)
        # Create a new state with updated analysis and messages
        new_state = state.copy()
        new_state["analysis"] = {
            "text": analysis,
            "chart": chart_data,
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "monthly_trend": monthly_trend,
            "timestamp": timestamp,
        }
        # Extract recommendation from analysis
        recommendation = "UNKNOWN"
        if "RECOMMENDATION: BUY" in analysis:
            recommendation = "BUY"
        elif "RECOMMENDATION: HOLD" in analysis:
            recommendation = "HOLD"
        elif "RECOMMENDATION: SELL" in analysis:
            recommendation = "SELL"

        # Save analysis to a JSON file
        analysis_record = {
            "timestamp": timestamp,
            "recommendation": recommendation,
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "monthly_trend": monthly_trend,
            "analysis_text": analysis,
            "chart_file": str(chart_filename),
        }

        analysis_file = HISTORY_DIR / f"bitcoin_analysis_{timestamp}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_record, f, indent=2)
        logger.info(f"Saved analysis to {analysis_file}")

        logger.info(f"Analysis complete. Recommendation: {recommendation}")

        # Add the result to messages
        analysis_content = f"""
        # Pool Range Analysis - {timestamp}

        **Market Data:**
        - Current Bitcoin Price: ${current_price:.2f}
        - 24-Hour Price Change: {price_change_24h:.2f}%
        - 30-Day Price Trend: {monthly_trend:.2f}%

        **Uniswap v4 Pool Details:**
        - Position Value: ${position_value:.2f}
        - USDT Allocation: ${usdt_value:.2f}
        - WBTC Allocation: {wbtc_value:.6f} WBTC
        - Lower Price Bound: ${min_value:.2f}
        - Higher Price Bound: ${max_value:.2f}
        
        {analysis}
        
        Chart saved to: {chart_filename}
        """

        new_messages = state.get("messages", []).copy()
        new_messages.append(AIMessage(content=analysis_content))
        new_state["messages"] = new_messages
        new_state["timestamp"] = timestamp

        return new_state
    except Exception as e:
        logger.error(f"Error generating analysis: {str(e)}")
        # Create a new state with error
        new_state = state.copy()
        new_state["error"] = f"Error generating analysis: {str(e)}"

        # Add error message
        new_messages = state.get("messages", []).copy()
        new_messages.append(AIMessage(content=f"Error generating analysis: {str(e)}"))
        new_state["messages"] = new_messages

        return new_state


def handle_error(state: AgentState) -> AgentState:
    """Handle errors in the workflow."""
    if state.get("error"):
        logger.error(f"Error in workflow: {state['error']}")
        new_state = state.copy()
        new_messages = state.get("messages", []).copy()
        new_messages.append(AIMessage(content=f"Error: {state['error']}"))
        new_state["messages"] = new_messages
        return new_state
    return state


def notify_user(state: AgentState) -> AgentState:
    """Notify user of new analysis (placeholder for notification system)."""
    if state.get("error"):
        return state

    # This is where you would implement notification logic
    # e.g., send email, push notification, Telegram message, etc.
    logger.info("Notifying user of new Bitcoin analysis...")

    # Example: Print to console (replace with your preferred notification method)
    if state.get("analysis"):
        analysis = state["analysis"]
        print("\n" + "=" * 50)
        print(f"NEW BITCOIN ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Price: ${analysis['current_price']}")
        print(f"24h Change: {analysis['price_change_24h']:.2f}%")
        print(f"30-Day Trend: {analysis['monthly_trend']:.2f}%")

        # Extract recommendation
        analysis_text = analysis["text"]
        if "RECOMMENDATION: BUY" in analysis_text:
            print("RECOMMENDATION: BUY")
        elif "RECOMMENDATION: HOLD" in analysis_text:
            print("RECOMMENDATION: HOLD")
        elif "RECOMMENDATION: SELL" in analysis_text:
            print("RECOMMENDATION: SELL")
        print("=" * 50 + "\n")

    return state


def compare_with_previous(state: AgentState) -> AgentState:
    """Compare current analysis with previous to detect significant changes."""
    if state.get("error"):
        return state

    try:
        # Get list of previous analysis files
        analysis_files = sorted(HISTORY_DIR.glob("bitcoin_analysis_*.json"))

        if len(analysis_files) < 2:
            logger.info("Not enough historical data for comparison")
            return state

        # Get the most recent previous analysis (excluding current one)
        prev_file = analysis_files[-2] if len(analysis_files) > 1 else None

        if prev_file:
            with open(prev_file, "r") as f:
                prev_analysis = json.load(f)

            current_analysis = state["analysis"]

            # Calculate price change between analyses
            price_change = (
                (current_analysis["current_price"] / prev_analysis["current_price"]) - 1
            ) * 100

            # Check if there's been a significant change
            if abs(price_change) >= 5:  # 5% threshold
                logger.info(f"Significant price change detected: {price_change:.2f}%")
                new_state = state.copy()
                new_messages = state.get("messages", []).copy()
                new_messages.append(
                    AIMessage(
                        content=f"""
                🚨 SIGNIFICANT PRICE CHANGE ALERT 🚨
                Price has changed by {price_change:.2f}% since last analysis.
                Previous price: ${prev_analysis["current_price"]}
                Current price: ${current_analysis["current_price"]}
                """
                    )
                )
                new_state["messages"] = new_messages
                return new_state
    except Exception as e:
        logger.error(f"Error comparing with previous analysis: {str(e)}")

    return state

# Build the graph with dict-based state
def build_bitcoin_agent():
    workflow = StateGraph(AgentState)


    
    # Add nodes
    workflow.add_node("get_current_price", get_current_bitcoin_price)
    workflow.add_node("get_historical_prices", get_historical_bitcoin_prices)
    workflow.add_node("generate_analysis", generate_analysis)
    workflow.add_node("compare_with_previous", compare_with_previous)
    workflow.add_node("notify_user", notify_user)
    workflow.add_node("handle_error", handle_error)

    # Add edges
    workflow.add_edge("get_current_price", "get_historical_prices")
    workflow.add_edge("get_historical_prices", "generate_analysis")
    workflow.add_edge("generate_analysis", "compare_with_previous")
    workflow.add_edge("compare_with_previous", "notify_user")
    workflow.add_edge("notify_user", END)
    workflow.add_edge("handle_error", END)

    # Set conditional edges
    workflow.add_conditional_edges(
        "get_current_price",
        lambda state: "handle_error" if state.get("error") else "get_historical_prices",
    )

    workflow.add_conditional_edges(
        "get_historical_prices",
        lambda state: "handle_error" if state.get("error") else "generate_analysis",
    )

    # Set entry point
    workflow.set_entry_point("get_current_price")

    # Compile the graph
    return workflow.compile()


# Function to run the agent
def run_bitcoin_agent(initial_query="Analyze Bitcoin price"):
    """Run the Bitcoin agent with the user query."""
    bitcoin_agent = build_bitcoin_agent()
    
    # Initialize state as a dictionary
    initial_state = {
        "messages": [HumanMessage(content=initial_query)],
        "current_price": None,
        "historical_prices": None,
        "analysis": None,
        "error": None,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    logger.info(f"Starting Bitcoin analysis run at {datetime.now()}")
    result = bitcoin_agent.invoke(initial_state)
    logger.info(f"Completed Bitcoin analysis run at {datetime.now()}")

    return result["messages"]


# Function to run the agent every 6 hours
def scheduled_analysis():
    """Run Bitcoin analysis on schedule."""
    logger.info("Running scheduled Bitcoin price analysis")
    messages = run_bitcoin_agent("Scheduled Bitcoin price analysis")

    # Log the latest message
    if messages:
        latest = messages[-1].content
        # logger.info(f"Analysis complete: {latest[:100]}...")
        logger.info(f"Analysis complete: {latest}...")


# Function to start the agent in autonomous mode
def start_autonomous_agent():
    """Start the Bitcoin price agent in autonomous 24/7 mode."""
    logger.info("Starting Bitcoin Price Agent in autonomous mode")
    logger.info("Agent will run every 6 hours")

    # Schedule the task to run every 6 hours
    schedule.every(6).hours.do(scheduled_analysis)

    # Also run immediately on startup
    scheduled_analysis()

    logger.info("Agent is running. Press Ctrl+C to stop.")

    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for scheduled tasks
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bitcoin Price Analysis Agent")
    parser.add_argument(
        "--mode",
        choices=["once", "autonomous"],
        default="once",
        help="Run once or in continuous autonomous mode",
    )
    args = parser.parse_args()

    if args.mode == "autonomous":
        start_autonomous_agent()
    else:
        # Run once
        result = run_bitcoin_agent("Analyze current Bitcoin price")
        for message in result:
            print(f"\n{message.type}: {message.content}")
