import numpy as np
from examples.carton_dimension.carton_dimension import BBox, CartonDimensionPipeline

class SequenceDetector:
    def __init__(self, values): self.values = iter(values)
    def detect(self, frame): return next(self.values)

class FixedRegressor:
    def __init__(self): self.seen = None
    def predict(self, features): self.seen = features; return (300., 200., 150.)

def test_collects_only_inside_frames_and_emits_on_exit():
    regressor = FixedRegressor()
    detector = SequenceDetector([[BBox(10, 10, 50, 30)], [BBox(20, 10, 70, 40)], [BBox(0, 10, 70, 40)]])
    pipeline = CartonDimensionPipeline(detector, regressor, border_margin=1, min_samples=2)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert pipeline.process(frame) is None
    assert pipeline.process(frame) is None
    result = pipeline.process(frame)
    assert result and (result.length, result.width, result.height, result.sampled_frames) == (300., 200., 150., 2)
    assert regressor.seen.shape == (9,)

def test_short_track_does_not_trigger_regression():
    pipeline = CartonDimensionPipeline(SequenceDetector([[BBox(10, 10, 20, 20)], []]), FixedRegressor(), min_samples=2)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert pipeline.process(frame) is None
    assert pipeline.process(frame) is None
