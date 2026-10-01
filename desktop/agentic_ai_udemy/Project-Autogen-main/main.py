import os
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

load_dotenv()

# llm_config = { "config_list": [{ "model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY") }] }

llm_config = {
    "config_list": [
        {
            "model": "gpt-5.6",  # This is your Azure deployment name
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_version": "2025-01-01-preview"
        }
    ]
}

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

user_proxy.initiate_chat(
    assistant,
    message= "Tell me a joke on data scientists"
)