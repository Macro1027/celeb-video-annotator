# Celebrity Video Annotator

A professional Python package for automatic face recognition and video annotation using state-of-the-art deep learning models.

## Features

- **Face Detection**: Uses MTCNN for robust face detection across video frames
- **Face Recognition**: Leverages deep learning embeddings for accurate face identification
- **Vector Database**: Powered by Pinecone for efficient similarity search
- **Video Annotation**: Automatically annotates videos with identified faces
- **Timeline Export**: Generate CSV timelines showing when each person appears
- **Batch Processing**: Efficient processing of large video files
- **Professional CLI**: Command-line interface for easy integration into workflows

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Install from source

```bash
git clone https://github.com/yourusername/celeb-video-annotator.git
cd celeb-video-annotator
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configuration

Create or update `config/config.yaml`:

```yaml
model_settings:
  batch_size: 48
  output_dir: "results/"
  video_path: "demo/your_video.mp4"
  api_key: "your_pinecone_api_key_here"
  index_name: "face-recognition-embeddings"
  target_label: "Person Name"  # Optional: highlight specific person
```

### 2. Prepare Dataset

Place your face recognition dataset in the following structure:

```
data/
├── Dataset.csv
└── Original Images/
    └── Original Images/
        ├── Person1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── Person2/
            ├── image1.jpg
            └── image2.jpg
```

### 3. Run the CLI

```bash
# Create face database and process video with timeline
python -m celeb_video_annotator.cli --create-database --generate-video --export-timeline

# Or use the installed command
celeb-video-annotator --create-database --generate-video --export-timeline
```

## Usage Examples

### Create Database Only
```bash
celeb-video-annotator --create-database
```

### Process Video Only
```bash
celeb-video-annotator --generate-video
```

### Full Pipeline with Timeline
```bash
celeb-video-annotator --create-database --generate-video --export-timeline
```

### Custom Config File
```bash
celeb-video-annotator --config path/to/custom/config.yaml --generate-video
```

## API Usage

```python
from celeb_video_annotator import AutomaticFaceRecognizer, load_config

# Load configuration
config = load_config('config/config.yaml')

# Initialize recognizer
recognizer = AutomaticFaceRecognizer(config)

# Process video
results = recognizer.extract_and_embeddings_from_video('video.mp4')

# Create annotated video
recognizer.rebuild_video_with_annotations(
    'video.mp4', 
    results, 
    'annotated_video.mp4'
)
```

## Project Structure

```
celeb-video-annotator/
├── celeb_video_annotator/        # Main package
│   ├── __init__.py
│   ├── cli.py                    # Command-line interface
│   ├── core/                     # Core functionality
│   │   ├── face_recognizer.py    # Main recognition logic
│   │   └── feature_extractor.py  # Feature extraction
│   ├── data/                     # Data handling
│   │   └── loader.py             # Dataset loading utilities
│   └── utils/                    # Utilities
│       └── config.py             # Configuration management
├── config/                       # Configuration files
│   └── config.yaml
├── data/                         # Dataset directory
├── tests/                        # Test files
├── scripts/                      # Utility scripts
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api_key` | Pinecone API key | Required |
| `batch_size` | Processing batch size | 48 |
| `output_dir` | Output directory | "results/" |
| `video_path` | Input video path | Required |
| `index_name` | Pinecone index name | "face-recognition-embeddings" |
| `target_label` | Specific person to highlight | None |

## Output Files

- **Recognition Results**: `{video_name}_recognition.json`
- **Annotated Video**: `{video_name}_annotated.mp4`
- **Timeline**: `{video_name}_timeline.csv`

## Requirements

- PyTorch >= 1.13.0
- OpenCV >= 4.8.0
- MTCNN for face detection
- Facenet-PyTorch for embeddings
- Pinecone for vector database
- See `requirements.txt` for complete list

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MTCNN for face detection
- FaceNet for face embeddings
- Pinecone for vector database infrastructure 