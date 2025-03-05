import time
from datetime import datetime
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# Define the state that will be passed between nodes
class AutomationState(TypedDict):
    current_time: str  # Store the current time as a string
    message: str       # Store the message to write
    task_completed: bool  # Track if the task is done

# Trigger node: Check if it's 5 PM
def check_time_trigger(state: AutomationState) -> AutomationState:
    # Get the current time
    now = datetime.now()
    state["current_time"] = now.strftime("%H:%M")  # Format as HH:MM (e.g., "17:00")
    
    # Check if it's 5 PM (17:00 in 24-hour format)
    if state["current_time"] == "10:21":
        state["message"] = "Time to relax"
    else:
        state["message"] = None  # No message if it's not 5 PM
    
    return state

# Action node: Write the message to a file
def write_to_file(state: AutomationState) -> AutomationState:
    if state["message"]:
        with open("automation_log.txt", "a") as f:
            f.write(f"{state['current_time']}: {state['message']}\n")
        state["task_completed"] = True
    else:
        state["task_completed"] = False  # Nothing to write if no message
    return state

# Conditional edge: Decide what to do based on the message
def route_task(state: AutomationState) -> Literal["write_to_file", "__end__"]:
    if state["message"]:
        return "write_to_file"  # Proceed to write if there's a message
    return "__end__"  # End the workflow if no message

# Build the graph
workflow = StateGraph(AutomationState)

# Add nodes
workflow.add_node("check_time", check_time_trigger)
workflow.add_node("write_to_file", write_to_file)

# Define the flow
workflow.set_entry_point("check_time")  # Start with checking the time
workflow.add_conditional_edges(
    "check_time",
    route_task,
    {
        "write_to_file": "write_to_file",  # Go to write action if message exists
        "__end__": END                     # End if no message
    }
)
workflow.add_edge("write_to_file", END)  # After writing, end the workflow

# Compile the graph
graph = workflow.compile()

# Run the graph
def run_automation():
    initial_state = {"current_time": "", "message": None, "task_completed": False}
    result = graph.invoke(initial_state)
    print("Workflow completed. Final state:", result)

# Simulate running it in a loop (for demo purposes)
if __name__ == "__main__":
    print("Running automation... (Checks every 5 seconds)")
    while True:
        run_automation()
        time.sleep(5)  # Check every 5 seconds (in reality, you'd adjust this)