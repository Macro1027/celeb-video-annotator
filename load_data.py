"""
Data loading utilities for the celebrity video annotator
"""
import os
import pandas as pd
from pathlib import Path
import zipfile


def load_dataset(csv_path: str = 'data/Dataset.csv') -> pd.DataFrame:
    """Load the face recognition dataset"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} entries")
    return df


def extract_dataset_zip(zip_path: str, extract_to: str = 'data/') -> None:
    """Extract the dataset zip file if needed"""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    Path(extract_to).mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Dataset extracted to {extract_to}")


def get_available_faces(images_dir: str = 'data/Original Images/Original Images') -> list:
    """Get list of available face directories"""
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    faces = [d for d in os.listdir(images_dir) 
             if os.path.isdir(os.path.join(images_dir, d))]
    
    print(f"Found {len(faces)} face directories: {faces}")
    return faces