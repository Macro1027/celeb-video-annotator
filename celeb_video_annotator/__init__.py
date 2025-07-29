"""
Celebrity Video Annotator - Face Recognition and Video Annotation Tool

A professional tool for automatic face recognition and video annotation
using MTCNN face detection and deep learning embeddings.
"""

__version__ = "1.0.0"
__author__ = "Marco Lee"
__email__ = "marcoleeml27@gmail.com"

# Lazy imports to avoid import errors when dependencies are missing
def _lazy_import():
    """Lazy import function to avoid immediate import errors"""
    try:
        from .core.face_recognizer import AutomaticFaceRecognizer
        from .core.feature_extractor import ExtractFeaturesMTCNN
        from .data.loader import load_dataset, get_available_faces
        from .utils.config import load_config, validate_config
        
        return {
            "AutomaticFaceRecognizer": AutomaticFaceRecognizer,
            "ExtractFeaturesMTCNN": ExtractFeaturesMTCNN,
            "load_dataset": load_dataset,
            "get_available_faces": get_available_faces,
            "load_config": load_config,
            "validate_config": validate_config
        }
    except ImportError as e:
        import warnings
        warnings.warn(f"Some dependencies are missing: {e}. Install requirements.txt for full functionality.")
        return {}

# Populate module namespace
_imports = _lazy_import()
globals().update(_imports)

__all__ = list(_imports.keys()) if _imports else [] 