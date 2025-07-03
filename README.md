# Crowd Management Project

This repository provides a simple people counting system using **YOLOv8**,
**Deep SORT** tracking and **FastAPI**. The application exposes a dashboard that
shows live counts of people entering and exiting a region from a video stream.
Redis is used to persist counts between restarts.

## Features

- Real‑time people detection using YOLOv8
- Object tracking with Deep SORT to avoid double counting
- Live video streaming and statistics via FastAPI
- WebSocket updates for the dashboard
- Editable settings stored in `config.json`

## Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

You also need a running Redis instance. The default configuration assumes it is
available on `localhost:6379`.

## Running

Start the application using:

```bash
python server.py
```

By default it binds to `http://0.0.0.0:8000`. Visit this address in your browser
to see the dashboard.

The video source defaults to the first webcam (`0`). You can provide another
source as a command‑line argument (e.g. an RTSP URL):

```bash
python server.py rtsp://example.com/stream
```

Settings such as maximum capacity can be adjusted from the `/settings` page of
the web interface.

## Configuration

The `config.json` file contains all tunable parameters such as detection model,
line position and frame rate. Modify this file or use the settings page to adapt
the system to your environment.

