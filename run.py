import logging
from app.app import create_app

logger = logging.getLogger(__name__)

app = create_app()

if __name__ == '__main__':
    logger.info("Starting Uvicorn server.")
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080, log_level="info")
    
