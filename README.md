# Celebrity Video Annotator & API

A powerful, containerized Python application that automatically recognizes and annotates celebrities in videos using deep learning, served via a high-performance FastAPI.

![Demo](https://user-images.githubusercontent.com/12345/your-demo-gif-url-here.gif) <!-- TODO: Add a GIF of the API in action -->

## ✨ Core Features

-   **🚀 High-Performance API**: Built with **FastAPI** for asynchronous, high-throughput video processing.
-   **📦 Containerized Deployment**: One-command launch with **Docker & Docker Compose**.
-   **asynchronous Job Queue**: Uses **Redis** to manage long-running annotation tasks.
-   **👨‍💻 State-of-the-Art ML**:
    -   **Face Detection**: Robust MTCNN from `facenet-pytorch`.
    -   **Face Recognition**: High-accuracy embeddings with FaceNet.
    -   **Vector Search**: Efficient similarity search powered by **Pinecone**.
-   **🎥 Automatic Annotation**: Overlays celebrity names directly onto the video.
-   **🔧 Flexible & Scalable**: Asynchronous architecture ready for scaling.
-   **💻 Versatile CLI**: Includes a command-line interface for local processing and database management.

---

## 🚀 Quick Start: Deploy with Docker

Get the API running in a few commands.

### Prerequisites
-   **Docker** and **Docker Compose** installed.
-   **Pinecone API Key**.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/celeb-video-annotator.git
cd celeb-video-annotator
```

### 2. Set Up Environment
Copy the example environment file and add your Pinecone API key.
```bash
cp env.example .env
```
Now, edit `.env` and add your key:
```dotenv
# .env
PINECONE_API_KEY=YOUR_PINECONE_API_KEY_HERE
```

### 3. Build and Launch the Services
This single command builds the Docker image, starts the API, and runs the Redis service.
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

### 4. Check Service Health
Verify that the API and its connection to Redis are healthy.
```bash
curl http://localhost:8000/health
```
You should see a response like:
```json
{
  "status": "ok",
  "checks": {
    "redis": "✅ Connected",
    "config": "✅ Loaded",
    "face_recognizer": "✅ Can import"
  }
}
```

---

## 📖 API Usage

Interact with the API using any HTTP client (like `curl`, Postman, or Python's `requests`).

### Interactive Docs
For a full, interactive API specification, go to:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

### 1. Annotate a Video
Send a video file to the `/annotate` endpoint. This will start a background job.

```bash
curl -X POST -F "file=@/path/to/your/video.mp4" "http://localhost:8000/annotate"
```

The API will immediately respond with a `job_id`:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "pending",
  "filename": "your_video.mp4",
  "file_size": 12345678,
  "submitted_at": "2023-10-27T10:00:00Z"
}
```

### 2. Check Job Status
Use the `job_id` to poll the `/jobs/{job_id}/status` endpoint.

```bash
curl http://localhost:8000/jobs/a1b2c3d4-e5f6-7890-1234-567890abcdef/status
```

The response will show the current status and progress:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "running", /* pending | running | completed | failed */
  "progress": 60.5,
  "download_url": null,
  "errors": []
}
```

### 3. Download Annotated Video
Once the job `status` is `"completed"`, a `download_url` will appear.

```bash
curl http://localhost:8000/jobs/a1b2c3d4-e5f6-7890-1234-567890abcdef/status
```
Response:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "completed",
  "progress": 100.0,
  "download_url": "/jobs/a1b2c3d4-e5f6-7890-1234-567890abcdef/download",
  "errors": []
}
```

Use the URL to download the final, annotated video file. The file will be saved in your `results/` directory.

---

##  CLI for Local Usage

For tasks like managing the face recognition database, a CLI is available.

### 1. Installation (without Docker)
```bash
# Clone the repo (if you haven't already)
git clone https://github.com/yourusername/celeb-video-annotator.git
cd celeb-video-annotator

# Install in editable mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Face Dataset
Your dataset of known faces should follow this structure:
```
data/
└── Original Images/
    └── Original Images/
        ├── Person1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── Person2/
            ├── image1.jpg
            └── image2.jpg
```

### 3. Create Face Database
Run this command to process the images and upload the face embeddings to Pinecone.
```bash
celeb-video-annotator --create-database
```
*Note: Ensure `PINECONE_API_KEY` is set as an environment variable or is present in `config/config.yaml`.*

### 4. Generate Video Locally
To process a video using the CLI without the API:
```bash
celeb-video-annotator --generate-video --config path/to/your/config.yaml
```

---

## 🏗️ Project Structure
```
celeb-video-annotator/
├── celeb_video_annotator/        # Main Python package
│   ├── api/                      # FastAPI service
│   │   ├── endpoints.py          # API endpoints and logic
│   │   └── schema.py             # Pydantic data models
│   ├── core/                     # Core ML functionality
│   │   ├── face_recognizer.py    # Main recognition logic
│   │   └── feature_extractor.py  # Feature extraction
│   ├── cli.py                    # Command-line interface
│   └── ...
├── config/                       # Configuration files
│   └── config.yaml
├── data/                         # Datasets for face recognition
├── results/                      # Output directory for annotated videos
├── temp/                         # Temporary storage for uploads
├── tests/                        # Unit and integration tests
├── .env                          # Local environment variables (gitignored)
├── env.example                   # Example environment file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup for installation
├── Dockerfile                    # Docker build instructions for the API
├── docker-compose.yml            # Defines API and Redis services
├── start_server.sh               # Server startup script (used by Docker)
└── README.md                     # This file
```

## ⚙️ Configuration
Configuration is handled via `config/config.yaml` and environment variables.

| Parameter | `config.yaml` | Environment Variable | Priority | Description |
|-----------|---------------|----------------------|----------|-------------|
| **Pinecone Key** | `api_key` | `PINECONE_API_KEY` | Env Var | **Required** for face database. |
| **Batch Size** | `batch_size` | | Config | Processing batch size for ML model. |
| **Output Dir** | `output_dir` | `OUTPUT_DIR` | Env Var | Where to save annotated videos. |
| **Index Name** | `index_name` | | Config | Pinecone index name. |

## 🛠️ Technologies Used
- **Python 3.12**
- **FastAPI**: Modern web framework
- **Uvicorn**: ASGI server
- **PyTorch**: Deep learning framework
- **Facenet-PyTorch**: Pre-trained models for face detection & recognition
- **OpenCV**: Video processing
- **Pinecone**: Vector database for similarity search
- **Redis**: In-memory database for job queueing
- **Docker & Docker Compose**: Containerization and service orchestration
- **Pydantic**: Data validation and settings management

## 🤝 Contributing
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

## 📄 License
This project is licensed under the MIT License. 
