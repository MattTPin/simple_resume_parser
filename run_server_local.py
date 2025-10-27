"""
Special function for running API endpoint server locally whenever required.

Simply run python run_serer_local.py in the terminal to launch the server
and begin hosting the swagger UI at `http://0.0.0.0:8001/docs`
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",       # points to your FastAPI app
        host="0.0.0.0",
        port=8001,
        reload=True             # auto-reload on code changes
    )