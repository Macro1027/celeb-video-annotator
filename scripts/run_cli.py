#!/usr/bin/env python3
"""
Convenience script to run the Celebrity Video Annotator CLI
"""
import sys
from pathlib import Path

# Add the parent directory to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from celeb_video_annotator.cli import main

if __name__ == "__main__":
    main() 