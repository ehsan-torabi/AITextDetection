"""Entrypoint for the Persian AI-Text Detection API (port 8000).

The app itself lives in aitextd.api; this launcher keeps the established
run command (`python server.py`) working.
"""

import uvicorn

from aitextd.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
