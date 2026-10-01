import os
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from dotenv import load_dotenv

load_dotenv()

# llm_config = { "config_list": [{ "model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY") }] }

llm_config = {
    "config_list": [
        {
            "model": "gpt-4",  # This is your Azure deployment name
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "base_url": os.getenv("AZURE_OPENAI_BASE_URL"),  # ✅ Note: ends with slash, no /deployments
            "api_type": "azure",
            "api_version": "2025-01-01-preview"
        }
    ]
}

maddy = ConversableAgent(
    "maddy",
    system_message = "Your name is Maddy and you are a part of a duo of comedians.",
    llm_config = llm_config,
    human_input_mode = "NEVER"
)

# # max consecutive auto reply
# joe = ConversableAgent(
#     "joe",
#     system_message = "Your name is Joe and you are a part of a duo of comedians.",
#     llm_config = llm_config,
#     human_input_mode = "NEVER",
#     max_consecutive_auto_reply=1
# )

joe = ConversableAgent(
    "joe",
    system_message = "Your name is Joe and you are a part of a duo of comedians.",
    llm_config = llm_config,
    human_input_mode = "NEVER",
    is_termination_msg = lambda msg: "terminate" in msg["content"].lower(),
)

joe.initiate_chat(maddy, message = "Maddy, tell me a joke and then say the TERMINATE ")