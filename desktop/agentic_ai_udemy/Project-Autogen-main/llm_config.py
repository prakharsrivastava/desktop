import os
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