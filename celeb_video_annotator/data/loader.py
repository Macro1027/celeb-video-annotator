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
    Uses KAGGLE_USERNAME and KAGGLE_KEY environment variables for authentication.
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
            print("2. Environment variables are set:")
            print("   - KAGGLE_USERNAME=your_username")
            print("   - KAGGLE_KEY=your_api_key")
            print("3. Or Kaggle API credentials are configured in ~/.kaggle/kaggle.json")
            print("4. You have accepted the dataset terms on Kaggle")
            raise FileNotFoundError(f"Could not obtain dataset: {csv_path}")
    
    # Check again after potential download
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file still not found after download attempt: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} entries")
    return df


def _download_kaggle_dataset() -> None:
    """Download the face recognition dataset from Kaggle using environment variables"""
    
    # Check if Kaggle credentials are available in environment
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    
    if kaggle_username and kaggle_key:
        print("Using Kaggle credentials from environment variables")
        _setup_kaggle_config_from_env(kaggle_username, kaggle_key)
    else:
        print("No Kaggle environment variables found, using existing ~/.kaggle/kaggle.json")
    
    try:
        # Run kaggle download command
        result = subprocess.run([
            'kaggle', 'datasets', 'download', 
            '-d', 'vasukipatel/face-recognition-dataset'
        ], capture_output=True, text=True, check=True)
        
        print("Dataset downloaded successfully from Kaggle")
        if result.stdout:
            print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Kaggle command failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        print("\nTroubleshooting tips:")
        print("1. Verify Kaggle credentials are correct")
        print("2. Check if you've accepted the dataset terms: https://www.kaggle.com/vasukipatel/face-recognition-dataset")
        print("3. Ensure dataset is still available at the URL")
        raise
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it with: pip install kaggle")
        raise


def _setup_kaggle_config_from_env(username: str, api_key: str) -> None:
    """Setup Kaggle configuration from environment variables"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_config = {
        "username": username,
        "key": api_key
    }
    
    kaggle_json_path = kaggle_dir / 'kaggle.json'
    import json
    with open(kaggle_json_path, 'w') as f:
        json.dump(kaggle_config, f)
    
    # Set proper permissions
    kaggle_json_path.chmod(0o600)
    print(f"✅ Kaggle credentials configured at {kaggle_json_path}")


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
    """Helper function to check and setup Kaggle credentials
    
    Checks both environment variables and ~/.kaggle/kaggle.json
    """
    # First check environment variables
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    
    if kaggle_username and kaggle_key:
        print("✅ Kaggle credentials found in environment variables")
        _setup_kaggle_config_from_env(kaggle_username, kaggle_key)
        return True
    
    # Then check kaggle.json file
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("❌ Kaggle credentials not found!")
        print("Please set up Kaggle API credentials using one of these methods:")
        print("\n🔧 Method 1: Environment Variables")
        print("  export KAGGLE_USERNAME=your_username")
        print("  export KAGGLE_KEY=your_api_key")
        print("\n🔧 Method 2: Kaggle JSON file")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Click 'Create New API Token'")
        print("  3. Place kaggle.json in ~/.kaggle/")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Check permissions
    import stat
    current_permissions = oct(os.stat(kaggle_json).st_mode)[-3:]
    if current_permissions != '600':
        print(f"⚠️  Warning: kaggle.json has permissions {current_permissions}, should be 600")
        print("Run: chmod 600 ~/.kaggle/kaggle.json")
    
    print("✅ Kaggle credentials found in ~/.kaggle/kaggle.json")
    return True


def check_environment_setup():
    """Check if all required environment variables and credentials are set up"""
    print("\n🔍 Environment Setup Check:")
    print("=" * 40)
    
    # Check Kaggle credentials
    kaggle_ok = setup_kaggle_credentials()
    
    # Check other environment variables
    env_vars = {
        'PINECONE_API_KEY': 'Pinecone API key for vector database',
        'VIDEO_PATH': 'Path to video file (optional if set in config)',
        'OUTPUT_DIR': 'Output directory (optional, defaults to results/)'
    }
    
    for var_name, description in env_vars.items():
        value = os.getenv(var_name)
        if value:
            print(f"✅ {var_name}: SET")
        else:
            print(f"⚪ {var_name}: not set ({description})")
    
    print("=" * 40)
    
    if not kaggle_ok:
        print("⚠️  Dataset download may fail without Kaggle credentials")
    
    return kaggle_ok 