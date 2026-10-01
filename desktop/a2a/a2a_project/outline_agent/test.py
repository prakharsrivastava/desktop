import os
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv
from outline_agent.agent_executor import create_foundry_agent_executor
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route


skills = [
    AgentSkill(
        id='book_flight',
        name='Flight Booking',
        description='Books a flight based on user preferences',
        tags=['travel', 'booking'],
        examples=[
            'Can you book a flight for me to New York?'
        ],
    ),
]


agent_card = AgentCard(
    name='AI Foundry Flight Booking Agent',
    description='An intelligent flight booking agent powered by Azure AI Foundry. '
    'I can help you book flights based on your preferences.',
    url=f'http://localhost:8000/',
    version='1.0.0',
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(streaming=True),
    skills= AgentSkill(
                id='book_flight',
                name='Flight Booking',
                description='Books a flight based on user preferences',
                tags=['travel', 'booking'],
                examples=[
                    'Can you book a flight for me to New York?'
                ],
            ),
)