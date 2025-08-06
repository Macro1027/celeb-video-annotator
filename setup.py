from setuptools import setup, find_packages

setup(
    name="celeb-video-annotator",
    version="1.0.0",
    description="Celebrity Video Annotator - Face Recognition API",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "pillow>=9.0.0",
        "tqdm>=4.64.0",
        "facenet-pytorch>=2.5.2",
        "mmcv>=2.0.0",
        "pinecone>=3.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        "redis>=5.0.0",
        "pyyaml>=6.0",
        "pathlib2>=2.3.7",
        "python-dotenv>=1.0.0",
        "kaggle>=1.5.12",
    ],
    entry_points={
        "console_scripts": [
            "celeb-video-annotator=celeb_video_annotator.cli:main",
        ],
    },
) 