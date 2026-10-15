"""Production construction shared by CLI and Discord jobs."""
from .models import Detector, Inpainter
from .balloons import BubbleSegmenter, TextEvidence
from .quality import QualityPipeline


def create_pipeline(settings):
    return QualityPipeline(settings, Detector(settings), Inpainter(settings), BubbleSegmenter(settings), TextEvidence(settings))
