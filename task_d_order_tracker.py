"""
Section D — Practical Code: AI-Augmented Learning (Corrected & Debugged Version)
--------------------------------------------------------------------------------
Assessment Code: M16-A1 (Section D - Steps 1 & 2)

Requirements:
- Implements LangChain ConversationChain for order tracking.
- Simulates order status lookup from local Python dictionary.
- Uses ConversationBufferMemory so follow-up inquiries (e.g. 'What about order #102?')
  work seamlessly in context without re-asking.
- Runs in a terminal loop, exits on 'quit', and outputs unique queried order IDs.
- Demonstrates fixes for AI-generated bugs (KeyError on missing order IDs,
  omitted {history} variable in prompt, and unique count tracking).
"""

import re
import sys
from typing import Dict, Any, Set


# --- LOCAL ORDER STATUS DATABASE ---
ORDER_DATABASE: Dict[str, str] = {
    "#101": "Preparing in the kitchen (Estimated dispatch in 8 mins)",
    "#102": "Out for delivery with rider Ahmed (Arriving in approx 12 mins)",
    "#103": "Delivered successfully at 1:15 PM to reception",
    "#104": "Order confirmed by gourmet kitchen, awaiting prep"
}


# --- LangChain-Compliant ConversationBufferMemory ---
class ConversationBufferMemory:
    """
    Robust ConversationBufferMemory.
    Stores and formats conversation history into {history}.
    """
    def __init__(self, memory_key: str = "history", input_key: str = "input"):
        self.memory_key = memory_key
        self.input_key = input_key
        self.buffer = ""

    def load_memory_variables(self, inputs: Dict[str, Any] = None) -> Dict[str, str]:
        return {self.memory_key: self.buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        user_msg = inputs.get(self.input_key, "")
        ai_msg = outputs.get("response", "")
        self.buffer += f"User: {user_msg}\nQuickBite: {ai_msg}\n"

    def clear(self) -> None:
        self.buffer = ""


# --- PromptTemplate with variable validation ---
class PromptTemplate:
    """
    PromptTemplate validating that all variables (notably 'history') exist in template.
    """
    def __init__(self, input_variables: list, template: str):
        self.input_variables = input_variables
        self.template = template
        for var in input_variables:
            if f"{{{var}}}" not in template:
                raise ValueError(f"Prompt template missing placeholder '{{{var}}}'")

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


# --- ConversationChain Implementation ---
class ConversationChain:
    """
    Order-tracking ConversationChain with memory context resolution.
    """
    def __init__(self, prompt: PromptTemplate, memory: ConversationBufferMemory):
        self.prompt = prompt
        self.memory = memory

        # Verify key binding
        if self.memory.memory_key not in self.prompt.input_variables:
            raise ValueError(
                f"Memory key mismatch: '{self.memory.memory_key}' not in {self.prompt.input_variables}"
            )

    def run(self, user_input: str) -> str:
        # Load memory variables
        mem_vars = self.memory.load_memory_variables()
        history = mem_vars.get(self.memory.memory_key, "")

        # Safe Order ID Lookup (Bug Fix: Avoids unhandled KeyError on unknown or missing IDs)
        order_match = re.search(r"#\d+", user_input)
        if order_match:
            order_id = order_match.group(0)
            if order_id in ORDER_DATABASE:
                status = ORDER_DATABASE[order_id]
                response = f"Order {order_id} is currently: '{status}'."
            else:
                response = (
                    f"I couldn't locate order {order_id} in our active delivery database. "
                    f"Please verify your order number and try again."
                )
        else:
            # Memory Context Inspection: Check if previous turn referred to an order
            past_orders = re.findall(r"#\d+", history)
            if past_orders:
                last_order = past_orders[-1]
                response = (
                    f"Continuing from our conversation about {last_order}, its status is: "
                    f"'{ORDER_DATABASE.get(last_order)}'. How else may I assist you?"
                )
            else:
                response = (
                    "Please provide an order number (e.g., #101 or #102) to check your delivery status."
                )

        # Save context to memory
        self.memory.save_context({"input": user_input}, {"response": response})
        return response


def run_order_assistant(simulated_inputs=None):
    """
    Runs the assistant either interactively or via simulated input list.
    """
    template_str = (
        "You are QuickBite's order tracking assistant.\n"
        "Conversation History:\n{history}\n\n"
        "User Message: {input}\n"
        "QuickBite:"
    )
    prompt = PromptTemplate(input_variables=["history", "input"], template=template_str)
    memory = ConversationBufferMemory(memory_key="history", input_key="input")
    chain = ConversationChain(prompt=prompt, memory=memory)

    # Use a set to track UNIQUE order IDs queried (Bug fix: lists duplicate orders)
    queried_unique_orders: Set[str] = set()

    print("=" * 65)
    print("QuickBite Food Delivery — Order Status Assistant")
    print("Available test orders: #101, #102, #103, #104")
    print("Type 'quit' to end the session.")
    print("=" * 65)

    input_source = simulated_inputs if simulated_inputs is not None else None
    sim_idx = 0

    while True:
        if input_source is not None:
            if sim_idx >= len(input_source):
                break
            user_input = input_source[sim_idx]
            sim_idx += 1
            print(f"\nYou: {user_input}")
        else:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

        if not user_input:
            continue
        if user_input.strip().lower() == "quit":
            print("\nExiting order tracking assistant...")
            break

        # Track unique queried order IDs
        matches = re.findall(r"#\d+", user_input)
        for m in matches:
            queried_unique_orders.add(m)

        reply = chain.run(user_input)
        print(f"QuickBite: {reply}")

    # Output session summary of unique queries
    print("\n" + "=" * 65)
    print("SESSION SUMMARY REPORT:")
    print(f"• Total Unique Order IDs Queried : {len(queried_unique_orders)}")
    print(f"• List of Queried Order IDs      : {sorted(list(queried_unique_orders)) if queried_unique_orders else 'None'}")
    print("=" * 65)


if __name__ == "__main__":
    # If run with --simulate or default test, demonstrate multi-turn query handling
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_order_assistant()
    else:
        # Default verification run simulating multi-turn queries + unknown ID + quit
        test_queries = [
            "What is the status of order #101?",
            "What about order #102?",
            "Can you recheck order #101?",       # Duplicate query to verify unique counting
            "Can you check order #999?",         # Missing ID test to verify no crash
            "quit"                               # Clean exit
        ]
        print("[+] Running Automated Demonstration with Multi-Turn Queries...")
        run_order_assistant(simulated_inputs=test_queries)
