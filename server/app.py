import uvicorn
from app.main import app as fastapi_app

# The validator looks for 'app' as the FastAPI instance
app = fastapi_app

def main():
    """
    OpenEnv Entry Point: The validator calls this function 
    to ensure the server can boot correctly.
    """
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    # This satisfies the 'missing if __name__ == "__main__"' requirement
    main()