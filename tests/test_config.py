"""Tests for configuration utilities"""
import pytest
import tempfile
import os
from pathlib import Path
from celeb_video_annotator.utils.config import load_config, validate_config


def test_load_config():
    """Test loading configuration from YAML file"""
    config_content = """
model_settings:
  batch_size: 48
  output_dir: "results/"
  video_path: "test.mp4"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        f.flush()
        
        try:
            config = load_config(f.name)
            assert 'model_settings' in config
            assert config['model_settings']['batch_size'] == 48
        finally:
            os.unlink(f.name)


def test_load_config_file_not_found():
    """Test error handling when config file doesn't exist"""
    with pytest.raises(FileNotFoundError):
        load_config('nonexistent_config.yaml')


def test_validate_config():
    """Test config validation"""
    config = {
        'output_dir': 'results/',
        'api_key': 'test_key'
    }
    
    # Should not raise any exception
    validate_config(config, ['output_dir', 'api_key'])


def test_validate_config_missing_fields():
    """Test config validation with missing fields"""
    config = {
        'output_dir': 'results/'
    }
    
    with pytest.raises(ValueError) as excinfo:
        validate_config(config, ['output_dir', 'api_key'])
    
    assert 'api_key' in str(excinfo.value) 