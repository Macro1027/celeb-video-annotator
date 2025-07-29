"""
Utility functions for the celebrity video annotator
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any

# Try to import python-dotenv for local development
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides
    
    Priority order:
    1. Environment variables (highest priority)
    2. .env file (if available and dotenv installed)
    3. YAML config file (lowest priority)
    """
    
    # Load .env file if available (for local development)
    if DOTENV_AVAILABLE:
        load_dotenv()
    
    # Load base configuration from YAML file
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file) or {}
    
    # Handle nested config structure
    if 'model_settings' in config:
        model_config = config['model_settings'].copy()
        # Preserve any root-level keys
        for key, value in config.items():
            if key != 'model_settings':
                model_config[key] = value
    else:
        model_config = config.copy()
    
    # Override with environment variables (highest priority)
    env_overrides = {
        'api_key': os.getenv('PINECONE_API_KEY'),
        'output_dir': os.getenv('OUTPUT_DIR'),
        'video_path': os.getenv('VIDEO_PATH'),
        'index_name': os.getenv('PINECONE_INDEX_NAME'),
        'target_label': os.getenv('TARGET_LABEL'),
        'batch_size': os.getenv('BATCH_SIZE'),
    }
    
    # Apply non-None environment overrides
    for key, value in env_overrides.items():
        if value is not None:
            # Convert batch_size to integer if provided
            if key == 'batch_size':
                try:
                    model_config[key] = int(value)
                except ValueError:
                    print(f"Warning: Invalid batch_size '{value}', using default")
            else:
                model_config[key] = value
    
    # Set default values if not provided anywhere
    defaults = {
        'output_dir': 'results/',
        'batch_size': 48,
        'index_name': 'face-recognition-embeddings'
    }
    
    for key, default_value in defaults.items():
        if key not in model_config or model_config[key] is None:
            model_config[key] = default_value
    
    # Debug: Print configuration source
    if os.getenv('DEBUG_CONFIG'):
        print("=== Configuration Sources ===")
        print(f"Config file: {config_path} ({'found' if os.path.exists(config_path) else 'not found'})")
        print(f"Environment variables: {[k for k, v in env_overrides.items() if v is not None]}")
        print("==============================")
    
    return model_config


def validate_config(config: Dict[str, Any], required_fields: list = None) -> None:
    """Validate configuration dictionary has required fields"""
    if required_fields is None:
        required_fields = ['output_dir']
    
    missing_fields = [field for field in required_fields if field not in config or config[field] is None]
    if missing_fields:
        print("\n🔧 Configuration Help:")
        print("Missing required configuration fields:", missing_fields)
        
        if 'api_key' in missing_fields:
            print("\nTo provide Pinecone API key:")
            print("  • Environment variable: export PINECONE_API_KEY=your_key")
            print("  • .env file: add PINECONE_API_KEY=your_key")
            print("  • config.yaml: add api_key under model_settings")
        
        if 'video_path' in missing_fields:
            print("\nTo provide video path:")
            print("  • Environment variable: export VIDEO_PATH=path/to/video.mp4")
            print("  • .env file: add VIDEO_PATH=path/to/video.mp4")
            print("  • config.yaml: add video_path under model_settings")
        
        raise ValueError(f"Missing required config fields: {missing_fields}")


def ensure_directory(directory_path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get basic video information"""
    import cv2
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_seconds': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def setup_kaggle_from_env():
    """Setup Kaggle credentials from environment variables"""
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    
    if not kaggle_username or not kaggle_key:
        print("⚠️  Kaggle credentials not found in environment variables")
        print("Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        return False
    
    # Create kaggle directory and config
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_config = {
        "username": kaggle_username,
        "key": kaggle_key
    }
    
    kaggle_json_path = kaggle_dir / 'kaggle.json'
    with open(kaggle_json_path, 'w') as f:
        import json
        json.dump(kaggle_config, f)
    
    # Set proper permissions
    kaggle_json_path.chmod(0o600)
    
    print("✅ Kaggle credentials configured from environment variables")
    return True


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the loaded configuration (without sensitive data)"""
    print("\n📋 Configuration Summary:")
    print("=" * 40)
    
    safe_config = config.copy()
    
    # Mask sensitive values
    if 'api_key' in safe_config and safe_config['api_key']:
        safe_config['api_key'] = f"{safe_config['api_key'][:8]}..." if len(safe_config['api_key']) > 8 else "***"
    
    for key, value in safe_config.items():
        if key not in ['api_key']:  # Don't print API key even if masked
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {'SET' if value else 'NOT SET'}")
    
    print("=" * 40) 