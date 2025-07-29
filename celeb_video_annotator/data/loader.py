"""
Data loading utilities for the celebrity video annotator
"""
import os
import subprocess
import pandas as pd
from pathlib import Path
import zipfile


def load_dataset(csv_path: str = 'data/Dataset.csv') -> pd.DataFrame:
    """Load the face recognition dataset
    
    If the CSV is not found, automatically downloads from Kaggle and extracts it.
    """
    if not os.path.exists(csv_path):
        print(f"Dataset file not found: {csv_path}")
        print("Attempting to download from Kaggle...")
        
        # Download from Kaggle
        try:
            _download_kaggle_dataset()
            # Extract if needed
            zip_path = 'face-recognition-dataset.zip'
            if os.path.exists(zip_path):
                print("Extracting dataset...")
                extract_dataset_zip(zip_path, 'data/')
            else:
                raise FileNotFoundError("Downloaded zip file not found")
                
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            print("Please ensure:")
            print("1. Kaggle CLI is installed: pip install kaggle")
            print("2. Kaggle API credentials are configured")
            print("3. You have accepted the dataset terms on Kaggle")
            raise FileNotFoundError(f"Could not obtain dataset: {csv_path}")
    
    # Check again after potential download
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file still not found after download attempt: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} entries")
    return df


def _download_kaggle_dataset() -> None:
    """Download the face recognition dataset from Kaggle"""
    try:
        # Run kaggle download command
        result = subprocess.run([
            'kaggle', 'datasets', 'download', 
            '-d', 'vasukipatel/face-recognition-dataset'
        ], capture_output=True, text=True, check=True)
        
        print("Dataset downloaded successfully from Kaggle")
        print(result.stdout if result.stdout else "Download completed")
        
    except subprocess.CalledProcessError as e:
        print(f"Kaggle command failed: {e}")
        print(f"Error output: {e.stderr}")
        raise
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it with: pip install kaggle")
        raise


def extract_dataset_zip(zip_path: str, extract_to: str = 'data/') -> None:
    """Extract the dataset zip file if needed"""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    Path(extract_to).mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Dataset extracted to {extract_to}")
    
    # Clean up zip file after extraction
    try:
        os.remove(zip_path)
        print(f"Cleaned up zip file: {zip_path}")
    except OSError as e:
        print(f"Could not remove zip file: {e}")


def get_available_faces(images_dir: str = 'data/Original Images/Original Images') -> list:
    """Get list of available face directories"""
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    faces = [d for d in os.listdir(images_dir) 
             if os.path.isdir(os.path.join(images_dir, d))]
    
    print(f"Found {len(faces)} face directories: {faces}")
    return faces


def verify_dataset_structure(base_dir: str = 'data/') -> bool:
    """Verify that the dataset has the expected structure"""
    required_files = [
        'Dataset.csv',
        'Original Images/Original Images'
    ]
    
    missing_items = []
    for item in required_files:
        full_path = os.path.join(base_dir, item)
        if not os.path.exists(full_path):
            missing_items.append(full_path)
    
    if missing_items:
        print(f"Missing dataset components: {missing_items}")
        return False
    
    print("Dataset structure verified successfully")
    return True


def setup_kaggle_credentials():
    """Helper function to check and setup Kaggle credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("Kaggle credentials not found!")
        print("Please set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Place kaggle.json in ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Check permissions
    import stat
    current_permissions = oct(os.stat(kaggle_json).st_mode)[-3:]
    if current_permissions != '600':
        print(f"Warning: kaggle.json has permissions {current_permissions}, should be 600")
        print("Run: chmod 600 ~/.kaggle/kaggle.json")
    
    return True 