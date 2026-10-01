"""start_server.py — Production server launcher."""
import os, uvicorn
from set_ssl_env import configure_ssl
configure_ssl()
uvicorn.run(
    "main:app",
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", "8001")),
    workers=int(os.environ.get("WORKERS", "2")),
    log_level=os.environ.get("LOG_LEVEL", "info").lower(),
)
