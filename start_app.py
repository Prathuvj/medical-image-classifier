import subprocess
import time
import os
import signal
import sys

def start():
    flask_process = subprocess.Popen(
        ["python", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(2)

    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app_streamlit.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        print("✅ Both Flask (API) and Streamlit (UI) are running...")
        print("🛑 Press Ctrl+C to stop.")
        flask_process.wait()
        streamlit_process.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")

        flask_process.terminate()
        streamlit_process.terminate()

        try:
            flask_process.wait(timeout=2)
            streamlit_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            flask_process.kill()
            streamlit_process.kill()

if __name__ == "__main__":
    start()