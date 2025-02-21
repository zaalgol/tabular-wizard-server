import asyncio
import logging
from app.app import create_app
import uvicorn

logger = logging.getLogger(__name__)

async def main():
    app = await create_app()  # Await the async app creation
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
