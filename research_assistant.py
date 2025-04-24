import operator
import random
from typing import TypedDict, Annotated, Sequence, Dict, Optional, List

from langgraph.graph import StateGraph, END

# --- 1. Define the State ---
# This dictionary holds the information passed between nodes.
# It acts as our short-term memory for a single run.

class ResearchState(TypedDict):
    user_query: str
    topic: Optional[str]
    # Simulating short-term memory within a run
    recalled_info: Optional[str]
    researched_info: Optional[str]
    draft_answer: Optional[str]
    final_answer: Optional[str]
    # For conditional logic
    evaluation_passed: bool
    # Adding a list to keep track of steps for clarity
    log: List[str]

# --- 2. Simulate Long-Term Memory ---
# We use a simple dictionary. In a real app, this would be a vector database.
long_term_memory_store: Dict[str, str] = {
    "langgraph basics": "LangGraph helps build stateful, multi-actor applications with LLMs.",
    "python syntax": "Python uses indentation for code blocks.",
}

# --- 3. Define Nodes (Simulated Agents / Functions) ---
# Each node takes the current state and returns a dictionary
# containing the updates to the state.

def parse_query_node(state: ResearchState) -> Dict[str, any]:
    """Extracts a simplified topic from the user query."""
    print("--- Node: Parsing Query ---")
    query = state['user_query']
    log = state.get('log', []) # Get existing log or start new list
    # Simple topic extraction (replace with more sophisticated NLP if needed)
    topic = query.lower().replace("what is", "").replace("tell me about", "").strip(" ?")
    log.append(f"Parsed query '{query}' into topic '{topic}'")
    return {"topic": topic, "log": log}

def recall_ltm_node(state: ResearchState) -> Dict[str, any]:
    """Checks the simulated LTM for relevant info."""
    print("--- Node: Recalling from LTM ---")
    topic = state.get('topic')
    log = state['log']
    recalled = None
    if topic:
        # Simple keyword matching for simulation
        for key, value in long_term_memory_store.items():
            if topic in key or key in topic:
                recalled = value
                log.append(f"Recalled from LTM for topic '{topic}': '{recalled}'")
                break
        if not recalled:
            log.append(f"No relevant info found in LTM for topic '{topic}'.")
    else:
        log.append("No topic identified to recall from LTM.")
    return {"recalled_info": recalled, "log": log}

def research_node(state: ResearchState) -> Dict[str, any]:
    """Simulates Agent 1: Performs research."""
    print("--- Node: Performing Research (Simulated Agent 1) ---")
    topic = state.get('topic')
    log = state['log']
    researched = None
    if topic:
        # Simulate finding some information
        possible_findings = [
            f"Detailed info on '{topic}': It involves complex concepts A, B, and C.",
            f"'{topic}' is often discussed in the context of X and Y.",
            f"A recent finding about '{topic}' suggests Z.",
        ]
        researched = random.choice(possible_findings)
        log.append(f"Researched info for topic '{topic}': '{researched}'")
    else:
        log.append("No topic identified to research.")
    return {"researched_info": researched, "log": log}

def synthesize_node(state: ResearchState) -> Dict[str, any]:
    """Simulates Agent 2: Combines info into a draft answer."""
    print("--- Node: Synthesizing Answer (Simulated Agent 2) ---")
    recalled = state.get('recalled_info')
    researched = state.get('researched_info')
    topic = state.get('topic', 'the topic')
    log = state['log']
    draft = f"Regarding {topic}:\n"
    if recalled:
        draft += f"- From past knowledge: {recalled}\n"
    if researched:
        draft += f"- New findings: {researched}\n"
    if not recalled and not researched:
        draft = f"Sorry, I couldn't find any information on '{topic}'."

    log.append(f"Synthesized draft answer.")
    return {"draft_answer": draft, "log": log}

def evaluate_node(state: ResearchState) -> Dict[str, any]:
    """Evaluates the draft answer (Simple check)."""
    print("--- Node: Evaluating Draft ---")
    draft = state.get('draft_answer', '')
    log = state['log']
    # Simple evaluation: check if it's longer than 50 chars and not the "sorry" message
    passed = len(draft) > 50 and "couldn't find any information" not in draft
    log.append(f"Evaluation result: {'Passed' if passed else 'Failed'}")
    return {"evaluation_passed": passed, "log": log}

def finalize_node(state: ResearchState) -> Dict[str, any]:
    """Finalizes the answer (if evaluation passed)."""
    print("--- Node: Finalizing Answer ---")
    draft = state['draft_answer']
    log = state['log']
    log.append("Finalizing the good answer.")
    # Could add more processing here if needed
    return {"final_answer": draft, "log": log}

def report_failure_node(state: ResearchState) -> Dict[str, any]:
    """Handles the case where evaluation failed."""
    print("--- Node: Reporting Failure ---")
    topic = state.get('topic', 'the topic')
    log = state['log']
    final_msg = f"Failed to generate a comprehensive answer for '{topic}'. More research might be needed."
    log.append("Reporting evaluation failure.")
    return {"final_answer": final_msg, "log": log}

def save_to_ltm_node(state: ResearchState) -> Dict[str, any]:
    """Saves the successful result to the simulated LTM."""
    print("--- Node: Saving to LTM ---")
    topic = state.get('topic')
    final_answer = state.get('final_answer')
    log = state['log']
    # Only save if we have a topic, a final answer, and it wasn't a failure message
    if topic and final_answer and "Failed to generate" not in final_answer:
         # Simple summary for LTM - in reality, more sophisticated
        summary_for_ltm = final_answer.split('\n')[1] if '\n' in final_answer else final_answer
        if summary_for_ltm.startswith("- "):
             summary_for_ltm = summary_for_ltm[2:] # Clean up bullet point

        # Prevent overwriting exactly identical info, slightly modify if topic exists
        if topic in long_term_memory_store and long_term_memory_store[topic] != summary_for_ltm:
             long_term_memory_store[topic + "_updated"] = summary_for_ltm # Avoid simple overwrite
             log.append(f"Updated LTM for topic '{topic}' (as '{topic}_updated').")
        elif topic not in long_term_memory_store:
             long_term_memory_store[topic] = summary_for_ltm
             log.append(f"Saved new info to LTM for topic '{topic}'.")
        else:
             log.append(f"Info for '{topic}' already in LTM and identical, not saving.")

    else:
        log.append("Skipping LTM save (no topic/answer or was failure).")
    # This node doesn't modify the main state elements for the flow, just the external LTM
    return {"log": log}


# --- 4. Define Conditional Logic ---

def should_finalize_or_report(state: ResearchState) -> str:
    """Determines the next step based on evaluation."""
    print("--- Condition: Checking Evaluation ---")
    if state.get("evaluation_passed", False):
        print("Outcome: Evaluation PASSED. Moving to Finalize.")
        return "finalize"
    else:
        print("Outcome: Evaluation FAILED. Moving to Report Failure.")
        return "report_failure"

# --- 5. Build the Graph ---

# Initialize the graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("parse_query", parse_query_node)
workflow.add_node("recall_ltm",     )
workflow.add_node("research", research_node)
workflow.add_node("synthesize", synthesize_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("finalize", finalize_node)
workflow.add_node("report_failure", report_failure_node)
workflow.add_node("save_to_ltm", save_to_ltm_node) # Add the LTM saving node

# Define the entry point
workflow.set_entry_point("parse_query")

# Add edges
workflow.add_edge("parse_query", "recall_ltm")
workflow.add_edge("recall_ltm", "research")
workflow.add_edge("research", "synthesize")
workflow.add_edge("synthesize", "evaluate")

# Add the conditional edge
workflow.add_conditional_edges(
    "evaluate",                          # Source node
    should_finalize_or_report,           # Function to determine the next node
    {
        "finalize": "finalize",          # If function returns "finalize", go to "finalize" node
        "report_failure": "report_failure" # If function returns "report_failure", go to "report_failure" node
    }
)

# Add edges from conditional branches towards the end or next steps
# Successful answers go to LTM saving before ending
workflow.add_edge("finalize", "save_to_ltm")
# Failed answers go directly to END
workflow.add_edge("report_failure", END)
# After attempting to save, the process ends
workflow.add_edge("save_to_ltm", END)


# Compile the graph into a runnable application
app = workflow.compile()

# --- 6. Run the Graph ---

print("--- Running Graph: First Query ---")
inputs1 = {"user_query": "Tell me about langgraph basics"}
final_state1 = app.invoke(inputs1)

print("\n--- Final State (Run 1) ---")
print(f"Query: {final_state1.get('user_query')}")
print(f"Topic: {final_state1.get('topic')}")
print(f"Final Answer: \n{final_state1.get('final_answer')}")
print("\nExecution Log (Run 1):")
for msg in final_state1.get('log', []):
    print(f"- {msg}")
print("\n--- LTM Store after Run 1 ---")
print(long_term_memory_store)


print("\n\n--- Running Graph: Second Query (Potentially using LTM from Run 1) ---")
inputs2 = {"user_query": "What is langgraph?"} # Similar topic, should hit LTM
final_state2 = app.invoke(inputs2)

print("\n--- Final State (Run 2) ---")
print(f"Query: {final_state2.get('user_query')}")
print(f"Topic: {final_state2.get('topic')}")
print(f"Final Answer: \n{final_state2.get('final_answer')}")
print("\nExecution Log (Run 2):")
for msg in final_state2.get('log', []):
    print(f"- {msg}")
print("\n--- LTM Store after Run 2 ---")
print(long_term_memory_store)


print("\n\n--- Running Graph: Third Query (New Topic) ---")
inputs3 = {"user_query": "Tell me about photosynthesis"} # New topic
final_state3 = app.invoke(inputs3)

print("\n--- Final State (Run 3) ---")
print(f"Query: {final_state3.get('user_query')}")
print(f"Topic: {final_state3.get('topic')}")
print(f"Final Answer: \n{final_state3.get('final_answer')}")
print("\nExecution Log (Run 3):")
for msg in final_state3.get('log', []):
    print(f"- {msg}")
print("\n--- LTM Store after Run 3 ---")
print(long_term_memory_store) # See if photosynthesis was added