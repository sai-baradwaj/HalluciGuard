import os

from dotenv import load_dotenv


load_dotenv()


API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError(
        "API_KEY is not configured. Please create a .env file."
    )