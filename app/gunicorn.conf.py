import os

workers = int(os.getenv("GUNICORN_WORKERS", "4"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "600"))
bind = f"0.0.0.0:{os.getenv('PORT', '8088')}"

worker_class = "uvicorn.workers.UvicornWorker"
