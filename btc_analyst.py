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

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bitcoin_agent.log"),
        logging.StreamHandler()
    ]
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

# Define functions for the nodes
def get_current_bitcoin_price(state: AgentState) -> AgentState:
    """Get current Bitcoin price from CoinGecko API."""
    try:
        logger.info("Fetching current Bitcoin price...")
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd", "include_market_cap": "true", 
                    "include_24hr_vol": "true", "include_24hr_change": "true"}
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
            params={
                "vs_currency": "usd",
                "from": start_timestamp,
                "to": end_timestamp
            }
        )
        response.raise_for_status()
        
        # Create a new state with updated historical_prices
        new_state = state.copy()
        new_state["historical_prices"] = response.json()
        logger.info(f"Retrieved {len(new_state['historical_prices']['prices'])} historical price points")
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
        price_data = pd.DataFrame(state["historical_prices"]["prices"], columns=["timestamp", "price"])
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
        
        # Craft prompt for the LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a cryptocurrency market analyst tasked with providing insights on Bitcoin.
            Based on the price data provided, determine if the recommendation should be BUY, HOLD, or SELL.
            Use the recent price performance, 24-hour change, and 30-day trend to justify your recommendation.
            Be brief but comprehensive and focus on current market conditions.
            End your analysis with a clear recommendation in bold format: **RECOMMENDATION: [BUY/HOLD/SELL]**"""),
            ("human", """Bitcoin Price Analysis:
            Current Price: ${current_price}
            24h Change: {price_change_24h:.2f}%
            30-Day Trend: {monthly_trend:.2f}%
            
            I need your analysis on whether this is a good time to BUY, HOLD, or SELL Bitcoin based on this data.
            Provide brief but insightful analysis and a clear recommendation.""")
        ])
        logger.info("prompt")
        logger.info("""Bitcoin Price Analysis:
            Current Price: ${current_price}
            24h Change: {price_change_24h:.2f}%
            30-Day Trend: {monthly_trend:.2f}%
            
            I need your analysis on whether this is a good time to BUY, HOLD, or SELL Bitcoin based on this data.
            Provide brief but insightful analysis and a clear recommendation.""")
        # Call the LLM
        logger.info("Calling Llama3.1:8b model for analysis...")
        llm = Ollama(model="llama3.1:8b")
        chain = prompt | llm | StrOutputParser()
        
        analysis = chain.invoke({
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "monthly_trend": monthly_trend
        })
        
        # Create a new state with updated analysis and messages
        new_state = state.copy()
        new_state["analysis"] = {
            "text": analysis,
            "chart": chart_data,
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "monthly_trend": monthly_trend,
            "timestamp": timestamp
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
            "chart_file": str(chart_filename)
        }
        
        analysis_file = HISTORY_DIR / f"bitcoin_analysis_{timestamp}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_record, f, indent=2)
        logger.info(f"Saved analysis to {analysis_file}")
        

        
        logger.info(f"Analysis complete. Recommendation: {recommendation}")
        
        # Add the result to messages
        analysis_content = f"""
        # Bitcoin Price Analysis - {timestamp}
        
        Current Price: ${current_price}
        24h Change: {price_change_24h:.2f}%
        30-Day Trend: {monthly_trend:.2f}%
        
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
        print("\n" + "="*50)
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
        print("="*50 + "\n")
    
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
            price_change = ((current_analysis["current_price"] / prev_analysis["current_price"]) - 1) * 100
            
            # Check if there's been a significant change
            if abs(price_change) >= 5:  # 5% threshold
                logger.info(f"Significant price change detected: {price_change:.2f}%")
                new_state = state.copy()
                new_messages = state.get("messages", []).copy()
                new_messages.append(AIMessage(content=f"""
                🚨 SIGNIFICANT PRICE CHANGE ALERT 🚨
                Price has changed by {price_change:.2f}% since last analysis.
                Previous price: ${prev_analysis["current_price"]}
                Current price: ${current_analysis["current_price"]}
                """))
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
        lambda state: "handle_error" if state.get("error") else "get_historical_prices"
    )
    
    workflow.add_conditional_edges(
        "get_historical_prices",
        lambda state: "handle_error" if state.get("error") else "generate_analysis"
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
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
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
        #logger.info(f"Analysis complete: {latest[:100]}...")
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
    parser.add_argument("--mode", choices=["once", "autonomous"], default="once", 
                        help="Run once or in continuous autonomous mode")
    args = parser.parse_args()
    
    if args.mode == "autonomous":
        start_autonomous_agent()
    else:
        # Run once
        result = run_bitcoin_agent("Analyze current Bitcoin price")
        for message in result:
            print(f"\n{message.type}: {message.content}")