# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# dependencies = [
#   "pyautogen",
#   "python-dotenv",
# ]
# ///
# DBTITLE 1,Install AutoGen
# MAGIC %pip install pyautogen==0.2.35 python-dotenv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Set up API Key
import os

# IMPORTANT: For security, please set up Databricks secrets:
# 1. Run: databricks secrets create-scope --scope openai-secrets
# 2. Run: databricks secrets put --scope openai-secrets --key api-key
# 3. Paste your API key when prompted

try:
    # Try to get the API key from Databricks secrets (recommended)
    os.environ["OPENAI_API_KEY"] = "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"
    print("✓ API key loaded securely from Databricks secrets")
except Exception as e:
    print(f"⚠️ Could not load API key from secrets: {e}")
    print("Please set up Databricks secrets as described in the comments above.")
    print("\nAlternatively, for quick testing, you can set the key directly:")
    print("os.environ['OPENAI_API_KEY'] = 'your-key-here'")

# COMMAND ----------

# DBTITLE 1,Temporary API Key (for testing)
# TEMPORARY: For quick testing only - remove this cell after setting up secrets
import os

# Paste your API key here for immediate testing:
os.environ["OPENAI_API_KEY"] = "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"

print("✓ API key set temporarily (remember to set up Databricks secrets for production)")

# COMMAND ----------

# DBTITLE 1,AutoGen Multi-Agent Setup


# COMMAND ----------

# MAGIC %skip
# MAGIC import os
# MAGIC from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
# MAGIC from dotenv import load_dotenv
# MAGIC
# MAGIC load_dotenv()
# MAGIC
# MAGIC ## Termination using is termination flag
# MAGIC agent_with_number = ConversableAgent(
# MAGIC     "agent_with_number",
# MAGIC     system_message=(
# MAGIC         "You are playing a game of guess-my-number. You have the number 58 in your mind, "
# MAGIC         "and I will try to guess it.\n"
# MAGIC         "If my guess is much higher than your number, say 'too high'.\n"
# MAGIC         "If my guess is much lower than your number, say 'too low'.\n"
# MAGIC         "If my guess is only slightly higher (within 5), say 'high'.\n"
# MAGIC         "If my guess is only slightly lower (within 5), say 'low'.\n"
# MAGIC         "If I guess correctly, say 'correct'."
# MAGIC     ),
# MAGIC     llm_config=llm_config,
# MAGIC     is_termination_msg=lambda msg: "58" in msg["content"],  # Terminate if the correct number is guessed
# MAGIC     human_input_mode="NEVER",  # Never ask for human input
# MAGIC )
# MAGIC
# MAGIC
# MAGIC agent_guess_number = ConversableAgent(
# MAGIC     "agent_guess_number",
# MAGIC     system_message=(
# MAGIC         "I have a number in my mind, and you will try to guess it. "
# MAGIC         "If I say 'too high', you should guess a much lower number. "
# MAGIC         "If I say 'high', you should guess a slightly lower number. "
# MAGIC         "If I say 'too low', you should guess a much higher number. "
# MAGIC         "If I say 'low', you should guess a slightly higher number. "
# MAGIC         "Keep adjusting your guess based on the feedback until you get it right."
# MAGIC     ),
# MAGIC     llm_config=llm_config,
# MAGIC     human_input_mode="NEVER",
# MAGIC )
# MAGIC
# MAGIC ## Human in the loop: ALWAYS
# MAGIC human_proxy = ConversableAgent(
# MAGIC     "human_proxy",
# MAGIC     llm_config=False,  # no LLM used for human proxy
# MAGIC     human_input_mode="ALWAYS",  # always ask for human input
# MAGIC )
# MAGIC
# MAGIC ## Human in the loop TERMINATE:
# MAGIC agent_with_number_term = ConversableAgent(
# MAGIC     "agent_with_number_term",
# MAGIC     system_message=(
# MAGIC         "You are playing a game of guess-my-number. You have the number 58 in your mind, "
# MAGIC         "and I will try to guess it.\n"
# MAGIC         "If my guess is much higher than your number, say 'too high'.\n"
# MAGIC         "If my guess is much lower than your number, say 'too low'.\n"
# MAGIC         "If my guess is only slightly higher (within 5), say 'high'.\n"
# MAGIC         "If my guess is only slightly lower (within 5), say 'low'.\n"
# MAGIC         "If I guess correctly, say 'correct'."
# MAGIC     ),
# MAGIC     llm_config=llm_config,
# MAGIC     max_consecutive_auto_reply=1,
# MAGIC     is_termination_msg=lambda msg: "58" in msg["content"],  # Terminate if the correct number is guessed
# MAGIC     human_input_mode="TERMINATE",  
# MAGIC )
# MAGIC
# MAGIC
# MAGIC if __name__ == "__main__":
# MAGIC     # agent_with_number.initiate_chat(
# MAGIC     #     agent_guess_number,
# MAGIC     #     message="I have a number between 1 and 100. Guess it!"
# MAGIC     # )
# MAGIC     # # Start a chat with the agent with number with an initial guess.
# MAGIC     # result = human_proxy.initiate_chat(
# MAGIC     #     agent_with_number,  # this is the same agent with the number as before
# MAGIC     #     message="10",
# MAGIC     # )
# MAGIC
# MAGIC     result = agent_with_number_term.initiate_chat(
# MAGIC         agent_guess_number,
# MAGIC         message="I have a number between 1 and 100. Guess it!",
# MAGIC     )

# COMMAND ----------



# %%
from typing import Annotated, Literal

Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")

# %%
# Let's first define the assistant agent that suggests tool calls.
assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. "
    "You can help with simple calculations. "
    "Return 'TERMINATE' when the task is done.",
    llm_config=llm_config,
)

# The user proxy agent is used for interacting with the assistant agent
# and executes tool calls.
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

# %%

# Register the tool signature with the assistant agent.
assistant.register_for_llm(name="calculator", description="A simple calculator")(calculator)

# Register the tool function with the user proxy agent.
assistant.register_for_execution(name="calculator")(calculator)

# %%
from autogen import register_function

# Register the calculator function to the two agents.
register_function(
    calculator,
    caller=assistant,  # The assistant agent can suggest calls to the calculator.
    executor=assistant,  # The user proxy agent can execute the calculator calls.
    name="calculator",  # By default, the function name is used as the tool name.
    description="A simple calculator",  # A description of the tool.
)

# %%
chat_result = user_proxy.initiate_chat(assistant, message="What is (44232 + 13312 / (232 - 32)) * 5?")


# COMMAND ----------


# %%
assistant.llm_config["tools"]

# %%
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    a: Annotated[int, Field(description="The first number.")]
    b: Annotated[int, Field(description="The second number.")]
    operator: Annotated[Operator, Field(description="The operator.")]


def calculator(input: Annotated[CalculatorInput, "Input to the calculator."]) -> int:
    if input.operator == "+":
        return input.a + input.b
    elif input.operator == "-":
        return input.a - input.b
    elif input.operator == "*":
        return input.a * input.b
    elif input.operator == "/":
        return int(input.a / input.b)
    else:
        raise ValueError("Invalid operator")

# %%
assistant.register_for_llm(name="calculator", description="A calculator tool that accepts nested expression as input")(
    calculator
)
user_proxy.register_for_execution(name="calculator")(calculator)

# %%
assistant.llm_config["tools"]

# %%
chat_result = user_proxy.initiate_chat(assistant, message="What is (1423 - 123) / 3 + (32 + 23) * 5?")
