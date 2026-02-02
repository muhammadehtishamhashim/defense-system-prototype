# Hifazat AI - Backend

The "Brain" of the Hifazat AI system. Built with FastAPI, it manages real-time monitoring, inference, and alert distribution.

## Features

-   **Multi-Model Inference**: Runs 3 concurrent detection loops (Threat, Theft, Border).
-   **WebSocket API**: Broadcasts real-time alerts to the frontend (`/ws`).
-   **Video Streaming**: MJPEG streams for live monitoring (`/video_feed/{source}`).
-   **Persistent Verification**: Manages snapshot lifecycle (Pending list -> Verified storage).

## Setup

1.  Create and activate a virtual environment:
    ```bash
    python -m venv new_env
    source new_env/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the server:
    ```bash
    python main.py
    ```
    Server will start at `http://0.0.0.0:8000`.

## API Endpoints

-   `GET /`: Health check ("Hifazat AI Brain is Active")
-   `GET /snapshots-list`: List all pending snapshots.
-   `POST /verify`: Verify an alert (moves file to `verified/`).
-   `POST /dismiss`: Delete an alert snapshot.
-   `GET /stats`: Get verified incident counts.
