#!/usr/bin/env python3
"""
Command line interface for the Celebrity Video Annotator
"""
import argparse
import sys
import os
from pathlib import Path


def create_database(config, selected_faces):
    """Create face database from dataset."""
    try:
        import pandas as pd
        from .core.face_recognizer import AutomaticFaceRecognizer
    except ImportError as e:
        print(f"Error: Missing required dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    print("Creating face database...")
    
    # Load and clean the DataFrame
    df = pd.read_csv('data/Dataset.csv')
    df_cleaned = df.loc[df['label'].isin(selected_faces)].sort_values(by=['label', 'id'])
    df_cleaned = df_cleaned.reset_index(drop=True)[['label', 'id']]
    
    print(f"Cleaned DataFrame with {len(df_cleaned)} entries:")
    print(df_cleaned.head())
    
    # Initialize recognizer with config dictionary
    recognizer = AutomaticFaceRecognizer(config)
    
    # Build database
    processed_df = recognizer.build_database_from_dataframe(df_cleaned, selected_faces)
    print(f"Database created with {len(processed_df)} face entries")
    
    return recognizer, processed_df


def process_video(config, recognizer=None):
    """Process video for face recognition."""
    try:
        from .core.face_recognizer import AutomaticFaceRecognizer
    except ImportError as e:
        print(f"Error: Missing required dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    if recognizer is None:
        print("Initializing face recognizer for video processing...")
        recognizer = AutomaticFaceRecognizer(config)
    
    video_path = config.get('video_path')
    if not video_path:
        print("Error: No video_path specified in config.yaml")
        return None, None
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None, None
    
    print(f"Processing video: {video_path}")
    
    # Extract and recognize faces from video
    results = recognizer.extract_and_embeddings_from_video(video_path)
    
    # Save recognition results
    base_name = Path(video_path).stem
    results_path = f"{config['output_dir']}/{base_name}_recognition.json"
    recognizer.save_results(results, results_path)
    
    print(f"Video processing completed! Processed {len(results)} frames")
    print(f"Results saved to: {results_path}")
    
    return recognizer, results


def create_annotated_video(config, recognizer, results):
    """Create annotated video with face recognition results."""
    video_path = config.get('video_path')
    base_name = Path(video_path).stem
    annotated_video_path = f"{config['output_dir']}/{base_name}_annotated.mp4"
    
    print("Creating annotated video...")
    recognizer.rebuild_video_with_annotations(
        video_path,
        results,
        annotated_video_path,
        font_scale=0.8,
        box_color=(0, 255, 0),  # Green boxes
        text_color=(255, 255, 255),  # White text
        box_thickness=3,
        min_confidence=0.8,
        only_label=config.get('target_label')  # Can be configured in yaml
    )
    
    print(f"Annotated video created: {annotated_video_path}")
    return annotated_video_path


def export_timeline(config, recognizer, results):
    """Export timeline CSV file."""
    video_path = config.get('video_path')
    base_name = Path(video_path).stem
    timeline_path = f"{config['output_dir']}/{base_name}_timeline.csv"
    
    print("Exporting timeline...")
    recognizer.export_timeline(results, timeline_path)
    print(f"Timeline exported to: {timeline_path}")
    return timeline_path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Celebrity Video Annotator - Face Recognition CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m celeb_video_annotator.cli --create-database --generate-video --export-timeline
  python -m celeb_video_annotator.cli --create-database
  python -m celeb_video_annotator.cli --generate-video --export-timeline
  python -m celeb_video_annotator.cli --generate-video --no-export-timeline

Environment Variables:
  PINECONE_API_KEY     Pinecone API key for vector database
  VIDEO_PATH          Path to video file to process
  OUTPUT_DIR          Output directory (default: results/)
  TARGET_LABEL        Specific person to highlight in annotations
  DEBUG_CONFIG        Show configuration loading details
        """
    )
    
    parser.add_argument(
        '--create-database',
        action='store_true',
        help='Create face database from dataset'
    )
    
    parser.add_argument(
        '--generate-video',
        action='store_true',
        help='Process video and generate annotated version (uses video_path from config.yaml or VIDEO_PATH env var)'  
    )
    
    parser.add_argument(
        '--export-timeline',
        action='store_true',
        default=False,
        help='Export timeline CSV file (default: False)'
    )
    
    parser.add_argument(
        '--no-export-timeline',
        action='store_true',
        help='Explicitly disable timeline export'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to config file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug information including configuration sources'
    )
    
    args = parser.parse_args()
    
    # Set debug environment variable if requested
    if args.debug:
        os.environ['DEBUG_CONFIG'] = '1'
    
    # Handle conflicting timeline options
    if args.no_export_timeline:
        args.export_timeline = False
    
    # Check if at least one action is specified
    if not (args.create_database or args.generate_video):
        parser.error("At least one action must be specified: --create-database or --generate-video")
    
    # Import utilities only when needed
    try:
        from .utils.config import load_config, validate_config, ensure_directory, print_config_summary
        from .data.loader import get_available_faces
    except ImportError as e:
        print(f"Error: Missing required dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Load configuration with environment variable support
    try:
        config = load_config(args.config)
        
        if args.debug:
            print_config_summary(config)
        
    except Exception as e:
        print(f"Error loading config file '{args.config}': {e}")
        sys.exit(1)
    
    # Validate required config fields
    required_fields = ['output_dir']
    if args.create_database or args.generate_video:
        # Only require api_key if PINECONE_API_KEY is not set in environment
        if not os.getenv('PINECONE_API_KEY'):
        required_fields.append('api_key')
    if args.generate_video:
        required_fields.append('video_path')
    
    try:
        validate_config(config, required_fields)
    except ValueError as e:
        print(f"Config validation error: {e}")
        sys.exit(1)
    
    # Create output directory
    ensure_directory(config['output_dir'])
    
    # Get selected faces
    try:
        selected_faces = get_available_faces()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        if args.create_database:
            print("Please ensure the dataset is available in the data/ directory.")
            selected_faces = []
        else:
            sys.exit(1)
    
    if selected_faces:
        print(f"Selected faces: {selected_faces}")
    
    recognizer = None
    results = None
    
    # Execute requested actions
    try:
        # Create database if requested
        if args.create_database:
            recognizer, processed_df = create_database(config, selected_faces)
        
        # Process video if requested
        if args.generate_video:
            recognizer, results = process_video(config, recognizer)
            
            if results is None:
                print("Video processing failed. Exiting.")
                sys.exit(1)
            
            # Create annotated video
            annotated_video_path = create_annotated_video(config, recognizer, results)
            
            # Export timeline if requested
            if args.export_timeline:
                timeline_path = export_timeline(config, recognizer, results)
        
        # Export timeline for existing results (if only timeline export is needed)
        elif args.export_timeline and not args.generate_video:
            print("Warning: --export-timeline specified without --generate-video")
            print("Timeline export requires video processing results.")
        
        print("\n" + "="*50)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        if args.create_database:
            print("✅ Face database created")
        
        if args.generate_video:
            print(f"✅ Video processed: {config.get('video_path')}")
            print(f"✅ Annotated video: {annotated_video_path}")
            
            if results:
                print(f"✅ Processed {len(results)} frames")
                
                # Debug results
                if recognizer:
                    recognizer.debug_results_data(results)
        
        if args.export_timeline and results:
            print(f"✅ Timeline exported: {timeline_path}")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 