import os
import json
import torch
import cv2
import mmcv
import numpy as np
import pandas as pd
import pinecone
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from collections import Counter, defaultdict
from PIL import Image
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
from ..core.feature_extractor import ExtractFeaturesMTCNN


class AutomaticFaceRecognizer:
    def __init__(self, config: Dict):
        """Initialize face recognizer with config dictionary"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract configuration values with defaults
        self.pinecone_api_key = config.get('api_key')
        if not self.pinecone_api_key:
            raise ValueError("api_key is required in config")
        
        self.index_name = config.get('index_name', 'face-recognition-embeddings')
        self.batch_size = config.get('batch_size', 48)
        self.output_dir = config.get('output_dir', 'results/')

        # Initialize models
        self._setup_models()
        self._setup_pinecone()

    # MAIN FUNCS
    def build_database_from_dataframe(self, df: pd.DataFrame, selected_faces: List[str] = None):
        """Build Pinecone database from existing DataFrame with 'label' and 'id' columns"""

        # Clean and filter the DataFrame
        if selected_faces:
            df_cleaned = df.loc[df['label'].isin(selected_faces)].sort_values(by=['label', 'id'])
        else:
            # If no selection provided, use all unique labels
            df_cleaned = df.sort_values(by=['label', 'id'])

        df_cleaned = df_cleaned.reset_index(drop=True)[['label', 'id']]

        print(f"Processing {len(df_cleaned)} face entries for {df_cleaned['label'].nunique()} people")

        # Process each person's data
        for person_name in df_cleaned['label'].unique():
            try:
                person_df = df_cleaned[df_cleaned['label'] == person_name].copy()

                # Generate embeddings for this person
                db_entry = self.net.align_and_embed_from_df(person_df, person_name)

                # Upload to Pinecone
                self.index.upsert(db_entry)

                print(f"Added {len(person_df)} entries for {person_name}")

                # Clean up memory
                del db_entry, person_df
                self._handle_cuda_memory()

            except Exception as e:
                print(f"Failed to process data for {person_name}: {e}")

        return df_cleaned


    def extract_and_embeddings_from_video(self, video_path: str) -> List[Dict]:
        """Process any video file and return face identification results"""
        video = mmcv.VideoReader(video_path)
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for frame in video]

        # Extract faces using your existing batch processing logic
        results = self._extract_faces_from_frames(frames)

        # Generate embeddings
        self._generate_embeddings(results)

        # Identify faces using Pinecone
        self._identify_faces(results)

        return results

    def _setup_models(self):
        """Initialize MTCNN and ResNet models"""
        self.mtcnn = MTCNN(margin=20, keep_all=True, post_process=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device).half() # FP 16 resnet instead of FP 32
        self.net = ExtractFeaturesMTCNN(mtcnn=self.mtcnn, resnet=self.resnet, device=self.device)
        print("Models initialized successfully")

    def _setup_pinecone(self):
        """Initialize Pinecone connection and index"""
        self.pc = pinecone.Pinecone(api_key=self.pinecone_api_key)

        # Create index if it doesn't exist
        if self.index_name not in [idx['name'] for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=512,
                metric="cosine",
                spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}}
            )

        self.index = self.pc.Index(self.index_name)

    def _create_dataframe_from_files(self, image_files: List[Path], person_name: str) -> pd.DataFrame:
        """Create a DataFrame from image files for a specific person"""
        data = []
        for img_path in image_files:
            data.append({
                'image_path': str(img_path),
                'person_id': person_name,
                'filename': img_path.name
            })

        return pd.DataFrame(data)

    def _process_images_by_filename(self, images_path: Path):
        """Process images where person names are encoded in filenames"""
        # Assumes filenames like "john_doe_001.jpg" or "jane-smith_photo.png"
        image_files = self._get_image_files(images_path)

        # Group files by person (extract person name from filename)
        person_files = {}
        for img_file in image_files:
            person_name = self._extract_person_from_filename(img_file.name)
            if person_name:
                if person_name not in person_files:
                    person_files[person_name] = []
                person_files[person_name].append(img_file)

        # Process each person's images
        for person_name, files in person_files.items():
            try:
                df = self._create_dataframe_from_files(files, person_name)
                db_entry = self.net.align_and_embed_from_df(df, person_name)
                self.index.upsert(db_entry)
                print(f"Added {len(files)} images for {person_name}")
            except Exception as e:
                print(f"Failed to process images for {person_name}: {e}")

    def _extract_person_from_filename(self, filename: str) -> Optional[str]:
        """Extract person name from filename using common naming conventions"""
        # Remove file extension
        name_part = os.path.splitext(filename)[0]

        # Try different separators and take the first part as person name
        for separator in ['_', '-', ' ']:
            if separator in name_part:
                # Take everything before the last separator that looks like a number
                parts = name_part.split(separator)
                # If last part is numeric, it's probably a sequence number
                if parts[-1].isdigit():
                    return separator.join(parts[:-1]).replace('_', ' ').replace('-', ' ').title()
                else:
                    return separator.join(parts).replace('_', ' ').replace('-', ' ').title()

        # If no separator, assume entire filename is person name
        return name_part.replace('_', ' ').replace('-', ' ').title()


    def _extract_faces_from_frames(self, frames: List) -> List[Dict]:
        """Extract faces from video frames using batch processing"""
        results = []
        buffer = []

        print(f"Processing {len(frames)} frames...")

        for global_idx, frame in tqdm(enumerate(frames), desc="Extracting faces"):
            buffer.append((global_idx, frame))

            # Process in batches for memory efficiency
            if (global_idx + 1) % self.batch_size == 0 or (global_idx + 1) == len(frames):
                try:
                    indices, batch_frames = zip(*buffer)

                    # Detect faces in batch
                    boxes_list, probs_list = self.mtcnn.detect(batch_frames)
                    faces_list = self.mtcnn(batch_frames)

                    # Process each frame in the batch
                    for local_idx in range(len(batch_frames)):
                        global_frame_idx = indices[local_idx]
                        boxes = boxes_list[local_idx] if boxes_list is not None else None
                        face = faces_list[local_idx]

                        if face is None:
                            # No face detected
                            results.append({
                                'frame_idx': global_frame_idx,
                                'boxes': None,
                                'face': None,
                                'embedding': None,
                                'confidence': None
                            })
                        elif face.ndim == 3:  # Single face
                            results.append({
                                'frame_idx': global_frame_idx,
                                'boxes': boxes,
                                'face': torch.unsqueeze(face, 0),
                                'embedding': None,
                                'confidence': probs_list[local_idx][0] if probs_list is not None else None
                            })
                        else:  # Multiple faces
                            for i, single_face in enumerate(face):
                                results.append({
                                    'frame_idx': global_frame_idx,
                                    'boxes': boxes[i] if boxes is not None and i < len(boxes) else None,
                                    'face': torch.unsqueeze(single_face, 0),
                                    'embedding': None,
                                    'confidence': probs_list[local_idx][i] if probs_list is not None else None
                                })

                    # Clean up GPU memory
                    del batch_frames, boxes_list, faces_list
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error processing batch starting at frame {indices[0]}: {e}")
                    # Add empty results for failed batch
                    for idx in indices:
                        results.append({
                            'frame_idx': idx,
                            'boxes': None,
                            'face': None,
                            'embedding': None,
                            'confidence': None
                        })

                buffer = []

        return results

    def _generate_embeddings(self, results: List[Dict], max_faces_per_frame=3,
                            confidence_threshold=0.7, max_total_faces=1000):
        """Generate embeddings with intelligent face sampling for better performance"""
        from collections import defaultdict
        import random

        # Filter for valid faces with confidence threshold
        valid_results = [
            r for r in results
            if r['face'] is not None and r.get('confidence', 1.0) >= confidence_threshold
        ]

        if not valid_results:
            print("No valid faces found to generate embeddings")
            return

        print(f"Found {len(valid_results)} valid faces above confidence threshold {confidence_threshold}")

        # Smart Sampling Strategy
        sampled_results = self._sample_faces_intelligently(
            valid_results, max_faces_per_frame, max_total_faces
        )

        print(f"Generating embeddings for {len(sampled_results)} sampled faces...")

        try:
            # Stack faces into tensor
            faces_tensor = torch.squeeze(
                torch.stack([r['face'] for r in sampled_results]),
                1
            ).to(self.device)

            # Generate embeddings in batches
            embeddings_list = []
            embedding_batch_size = 64  # Increased for efficiency

            with torch.no_grad():
                for i in tqdm(range(0, len(faces_tensor), embedding_batch_size),
                            desc="Computing embeddings"):
                    batch_end = min(i + embedding_batch_size, len(faces_tensor))
                    batch_faces = faces_tensor[i:batch_end]

                    # Use half precision for faster processing
                    if self.device.type == 'cuda':
                        batch_faces = batch_faces.half()

                    batch_embeddings = self.resnet(batch_faces.half()).float()
                    embeddings_list.append(batch_embeddings)

            # Concatenate all embeddings
            all_embeddings = torch.cat(embeddings_list, dim=0).cpu()

            # Assign embeddings back to sampled results
            for r, emb in zip(sampled_results, all_embeddings):
                r['embedding'] = emb

            # Set embeddings to None for non-sampled results
            sampled_indices = {id(r) for r in sampled_results}
            for r in valid_results:
                if id(r) not in sampled_indices:
                    r['embedding'] = None

            del faces_tensor, all_embeddings, embeddings_list
            torch.cuda.empty_cache()

            print(f"Embedding generation complete for {len(sampled_results)} faces")

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            for r in valid_results:
                r['embedding'] = None

    def _sample_faces_intelligently(self, valid_results, max_faces_per_frame, max_total_faces):
        """Sample faces intelligently to get good representation while managing compute"""
        from collections import defaultdict
        import random

        # Group faces by frame
        faces_by_frame = defaultdict(list)
        for r in valid_results:
            faces_by_frame[r['frame_idx']].append(r)

        sampled_results = []

        # Limit faces per frame
        for frame_idx, frame_faces in faces_by_frame.items():
            if len(frame_faces) <= max_faces_per_frame:
                sampled_results.extend(frame_faces)
            else:
                # Sort by confidence and take top faces
                sorted_faces = sorted(frame_faces,
                                    key=lambda x: x.get('confidence', 0),
                                    reverse=True)
                sampled_results.extend(sorted_faces[:max_faces_per_frame])

        return sampled_results

    def _identify_faces(self, results: List[Dict]):
        """Identify faces using Pinecone similarity search"""
        print("Identifying faces using Pinecone database...")

        for i, r in tqdm(enumerate(results), desc="Identifying faces"):
            if r['embedding'] is None:
                r['label'] = None
                r['similarity_score'] = None
            else:
                try:
                    # Convert embedding to list for Pinecone query
                    query_embedding = r['embedding'].cpu().numpy().tolist()

                    # Query Pinecone for similar embeddings
                    query_result = self.index.query(
                        vector=query_embedding,
                        top_k=10,
                        include_metadata=True
                    )

                    similar_matches = query_result.get('matches', [])

                    if similar_matches:
                        # Count votes for each person
                        person_ids = [match['metadata']['person_id'] for match in similar_matches]
                        counter = Counter(person_ids)

                        # Get the most common person and their confidence
                        most_common = counter.most_common(1)[0]
                        r['label'] = most_common[0]
                        r['vote_count'] = most_common[1]
                        r['similarity_score'] = similar_matches[0]['score']  # Best match score

                        # Calculate confidence based on vote ratio
                        total_votes = sum(counter.values())
                        r['confidence_ratio'] = most_common[1] / total_votes
                    else:
                        r['label'] = 'Unknown'
                        r['similarity_score'] = 0.0
                        r['vote_count'] = 0
                        r['confidence_ratio'] = 0.0

                except Exception as e:
                    print(f"Error identifying face in frame {r['frame_idx']}: {e}")
                    r['label'] = 'Error'
                    r['similarity_score'] = None
                    r['vote_count'] = 0
                    r['confidence_ratio'] = 0.0

    def save_results(self, results: List[Dict], output_path: str):
        """Save recognition results to JSON file with comprehensive serialization handling"""
        from copy import deepcopy

        def clean_json_serializable(data):
            """Recursively convert all non-JSON-serializable types to native Python types"""
            if isinstance(data, dict):
                return {k: clean_json_serializable(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_json_serializable(item) for item in data]
            elif isinstance(data, tuple):
                return [clean_json_serializable(item) for item in data]
            # Handle NumPy scalar types
            elif isinstance(data, (np.float32, np.float64)):
                return float(data)
            elif isinstance(data, (np.int32, np.int64, np.uint32, np.uint64)):
                return int(data)
            elif isinstance(data, np.bool_):
                return bool(data)
            # Handle NumPy arrays
            elif isinstance(data, np.ndarray):
                return data.tolist()
            # Handle PyTorch tensors
            elif isinstance(data, torch.Tensor):
                return data.cpu().numpy().tolist()
            # Handle None and native Python types
            else:
                return data

        # Create a deep copy and clean all data
        print("Converting data for JSON serialization...")
        serializable_results = []

        for r in results:
            clean_result = clean_json_serializable(r.copy())

            # Remove face tensors to reduce file size (they're too large for JSON)
            if 'face' in clean_result:
                clean_result['face'] = None

            serializable_results.append(clean_result)

        # Generate summary statistics
        summary = self._generate_summary_stats(results)
        clean_summary = clean_json_serializable(summary)

        # Prepare final output structure
        output_data = {
            'summary': clean_summary,
            'results': serializable_results,
            'metadata': {
                'total_frames': len(results),
                'processing_date': str(pd.Timestamp.now()),
                'model_info': {
                    'mtcnn_margin': 20,
                    'resnet_model': 'vggface2',
                    'pinecone_index': self.index_name
                }
            }
        }

        # Final cleaning pass on the entire output structure
        output_data = clean_json_serializable(output_data)

        # Save to JSON with error handling
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Results saved successfully to {output_path}")
            print(f"Summary: {clean_summary}")
        except Exception as e:
            print(f"Unexpected error saving results: {e}")
            raise

    def _generate_summary_stats(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from results"""
        total_frames = len(results)
        frames_with_faces = len([r for r in results if r['face'] is not None])
        identified_faces = len([r for r in results if r['label'] is not None and r['label'] not in ['Unknown', 'Error']])

        # Count identifications by person
        person_counts = Counter()
        for r in results:
            if r['label'] and r['label'] not in ['Unknown', 'Error', None]:
                person_counts[r['label']] += 1

        return {
            'total_frames': total_frames,
            'frames_with_faces': frames_with_faces,
            'identified_faces': identified_faces,
            'face_detection_rate': frames_with_faces / total_frames if total_frames > 0 else 0,
            'identification_rate': identified_faces / frames_with_faces if frames_with_faces > 0 else 0,
            'people_identified': dict(person_counts.most_common())
        }

    def export_timeline(self, results: List[Dict], output_path: str):
        """Export a timeline showing when each person appears in the video"""
        timeline_data = []

        for r in results:
            if r['label'] and r['label'] not in ['Unknown', 'Error', None]:
                timeline_data.append({
                    'frame': r['frame_idx'],
                    'person': r['label'],
                    'confidence': r.get('confidence_ratio', 0),
                    'similarity_score': r.get('similarity_score', 0)
                })

        timeline_df = pd.DataFrame(timeline_data)
        timeline_df.to_csv(output_path, index=False)
        print(f"Timeline exported to {output_path}")

        return timeline_df

    def _handle_cuda_memory(self):
        """Clean up CUDA memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _validate_dataframe_and_video(self, df: pd.DataFrame, video_path: str):
        """Validate DataFrame structure and video path"""

        # Check DataFrame has required columns
        required_columns = ['label', 'id']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

        # Check DataFrame is not empty
        if len(df) == 0:
            raise ValueError("DataFrame is empty")

        # Check video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Display DataFrame info
        print(f"DataFrame contains {len(df)} face entries")
        print(f"Unique people in dataset: {df['label'].unique().tolist()}")
        print(f"Entries per person: {df['label'].value_counts().to_dict()}")

        # Check CUDA availability
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU (will be slower)")

    def rebuild_video_with_annotations(self, video_path: str, results: List[Dict], output_path: str,
                                 font_scale: float = 0.7, box_color: tuple = (0, 255, 0),
                                 text_color: tuple = (255, 255, 255), box_thickness: int = 2, min_confidence=0.5, only_label=None):
        import cv2
        from collections import defaultdict

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height} @ {fps} fps, {total_frames} frames")

        results_by_frame = defaultdict(list)
        valid_detections = 0

        for result in results:
            frame_idx = result['frame_idx']
            results_by_frame[frame_idx].append(result)

            # Count valid detections for debugging
            if result.get('boxes') is not None:
                valid_detections += 1

        print(f"Total results: {len(results)}")
        print(f"Results with valid boxes: {valid_detections}")
        print(f"Frames with detections: {len(results_by_frame)}")

        fourcc_options = [
            cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'XVID'),
            cv2.VideoWriter_fourcc(*'MJPG')
        ]

        out = None
        for fourcc in fourcc_options:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {fourcc}")
                break
            out.release()

        if not out or not out.isOpened():
            raise RuntimeError("Could not initialize video writer")

        frame_count = 0
        frames_annotated = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for current frame
            frame_results = results_by_frame.get(frame_count, [])

            # **FIX 4: Apply annotations (modified drawing logic)**
            annotated_frame = self._draw_annotations_on_frame_fixed(
                frame, frame_results, font_scale, box_color, text_color, box_thickness, min_confidence, only_label
            )

            # Count annotated frames
            if frame_results:
                frames_annotated += 1
                if frames_annotated <= 5:  # Log first 5 annotated frames
                    print(f"Frame {frame_count}: Added {len(frame_results)} annotations")

            # Write frame
            out.write(annotated_frame)

            frame_count += 1

            # Progress update
            if frame_count % 500 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Annotated {frames_annotated} frames so far")

        cap.release()
        out.release()

        print(f"Video processing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with annotations: {frames_annotated}")
        print(f"Output saved to: {output_path}")


    def _draw_annotations_on_frame_fixed(self, frame, frame_results, font_scale, box_color, text_color, box_thickness, min_confidence=0.5, only_label=None):
        """Fixed annotation drawing with proper filtering logic"""
        import cv2

        annotated_frame = frame.copy()

        for result in frame_results:
            boxes = result.get('boxes')

            if boxes is not None:
                try:
                    # **FIX 1: Get confidence_ratio FIRST**
                    confidence_ratio = result.get('confidence_ratio', 0)
                    label = result.get('label')

                    # **FIX 2: Filter by label EARLY (most efficient)**
                    if only_label is not None and label != only_label:
                        continue

                    # **FIX 3: Filter by confidence EARLY**
                    if confidence_ratio < min_confidence:
                        continue

                    # Only do expensive coordinate validation for valid candidates
                    if len(boxes) == 4:
                        x1, y1, x2, y2 = int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])

                        # Basic coordinate validation
                        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:

                            # Determine display text and color
                            if label and label not in [None, 'Unknown', 'Error']:
                                display_text = f"{label} ({confidence_ratio:.2f})"
                                color = box_color  # Green for identified faces
                            else:
                                display_text = "Face Detected"
                                color = (0, 255, 255)  # Yellow for unidentified faces

                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness)

                            # Draw text label
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, 1)

                            # Text position
                            text_x = x1
                            text_y = y1 - 10 if y1 > 30 else y1 + text_height + 15

                            # Text background
                            cv2.rectangle(annotated_frame,
                                        (text_x, text_y - text_height - baseline),
                                        (text_x + text_width, text_y + baseline),
                                        color, -1)

                            cv2.putText(annotated_frame, display_text, (text_x, text_y),
                                    font, font_scale, text_color, 1, cv2.LINE_AA)

                except Exception as e:
                    print(f"Error drawing annotation: {e}")
                    continue

        return annotated_frame

    def debug_results_data(self, results):
        """Debug the results to see what's actually in there"""
        print(f"Total results: {len(results)}")

        # Count different types of results
        no_boxes = sum(1 for r in results if r.get('boxes') is None)
        valid_labels = sum(1 for r in results if r.get('label') not in [None, 'Unknown', 'Error'])
        has_faces = sum(1 for r in results if r.get('face') is not None)

        print(f"Results with no boxes: {no_boxes}")
        print(f"Results with valid labels: {valid_labels}")
        print(f"Results with faces detected: {has_faces}")

        # Show first few results with boxes
        with_boxes = [r for r in results if r.get('boxes') is not None][:5]
        for i, r in enumerate(with_boxes):
            print(f"Sample {i}: frame={r['frame_idx']}, boxes={r['boxes']}, label={r.get('label')}") 