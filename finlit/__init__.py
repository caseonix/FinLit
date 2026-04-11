"""FinLit - Canadian Financial Document Intelligence Framework."""
from finlit import schemas
from finlit.extractors.base_vision import BaseVisionExtractor
from finlit.extractors.vision_extractor import VisionExtractor
from finlit.pipeline import BatchPipeline, DocumentPipeline
from finlit.result import ExtractionResult
from finlit.schema import Field, Schema

__version__ = "0.3.0"

__all__ = [
    "DocumentPipeline",
    "BatchPipeline",
    "Schema",
    "Field",
    "ExtractionResult",
    "schemas",
    "VisionExtractor",
    "BaseVisionExtractor",
]
