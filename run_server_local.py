"""
Special function for running API endpoint server locally whenever required.

Simply run python run_serer_local.py in the terminal to launch the server
and begin hosting the swagger UI at `http://0.0.0.0:8001/docs`
"""
import uvicorn
import signal
import sys

def main():
    # Use Uvicorn programmatically for proper cleanup on Ctrl+C
    config = uvicorn.Config(
        "api.server:app",       # points to your FastAPI app
        host="0.0.0.0",
        port=8001,
        reload=True
    )
    server = uvicorn.Server(config)

    def handle_exit(sig, frame):
        print("\nShutting down gracefully...")
        # This triggers Uvicorn's graceful shutdown
        server.should_exit = True

    # Register signal handlers for graceful exit
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    server.run()
    print("Server stopped cleanly.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")
        sys.exit(0)