import os
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from llm_config import llm_config
from dotenv import load_dotenv

load_dotenv()

## Termination using is termination flag
agent_with_number = ConversableAgent(
    "agent_with_number",
    system_message=(
        "You are playing a game of guess-my-number. You have the number 58 in your mind, "
        "and I will try to guess it.\n"
        "If my guess is much higher than your number, say 'too high'.\n"
        "If my guess is much lower than your number, say 'too low'.\n"
        "If my guess is only slightly higher (within 5), say 'high'.\n"
        "If my guess is only slightly lower (within 5), say 'low'.\n"
        "If I guess correctly, say 'correct'."
    ),
    llm_config=llm_config,
    is_termination_msg=lambda msg: "58" in msg["content"],  # Terminate if the correct number is guessed
    human_input_mode="NEVER",  # Never ask for human input
)


agent_guess_number = ConversableAgent(
    "agent_guess_number",
    system_message=(
        "I have a number in my mind, and you will try to guess it. "
        "If I say 'too high', you should guess a much lower number. "
        "If I say 'high', you should guess a slightly lower number. "
        "If I say 'too low', you should guess a much higher number. "
        "If I say 'low', you should guess a slightly higher number. "
        "Keep adjusting your guess based on the feedback until you get it right."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

## Human in the loop: ALWAYS
human_proxy = ConversableAgent(
    "human_proxy",
    llm_config=False,  # no LLM used for human proxy
    human_input_mode="ALWAYS",  # always ask for human input
)

## Human in the loop TERMINATE:
agent_with_number_term = ConversableAgent(
    "agent_with_number_term",
    system_message=(
        "You are playing a game of guess-my-number. You have the number 58 in your mind, "
        "and I will try to guess it.\n"
        "If my guess is much higher than your number, say 'too high'.\n"
        "If my guess is much lower than your number, say 'too low'.\n"
        "If my guess is only slightly higher (within 5), say 'high'.\n"
        "If my guess is only slightly lower (within 5), say 'low'.\n"
        "If I guess correctly, say 'correct'."
    ),
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    is_termination_msg=lambda msg: "58" in msg["content"],  # Terminate if the correct number is guessed
    human_input_mode="TERMINATE",  
)


if __name__ == "__main__":
    # agent_with_number.initiate_chat(
    #     agent_guess_number,
    #     message="I have a number between 1 and 100. Guess it!"
    # )
    # # Start a chat with the agent with number with an initial guess.
    # result = human_proxy.initiate_chat(
    #     agent_with_number,  # this is the same agent with the number as before
    #     message="10",
    # )

    result = agent_with_number_term.initiate_chat(
        agent_guess_number,
        message="I have a number between 1 and 100. Guess it!",
    )