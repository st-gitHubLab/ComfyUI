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


def test_flush_predicts_carton_at_end_of_video():
    regressor = FixedRegressor()
    pipeline = CartonDimensionPipeline(SequenceDetector([[BBox(10, 10, 30, 30)], [BBox(12, 10, 34, 30)]]), regressor, min_samples=2)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline.process(frame)
    pipeline.process(frame)
    result = pipeline.flush()
    assert result is not None
    assert result.sampled_frames == 2
    assert not pipeline.collecting


def test_detection_record_contains_bbox_and_confidence():
    from examples.carton_dimension.carton_dimension import serialize_detections

    record = serialize_detections(4, [BBox(1, 2, 11, 22, 0.87)], 0, "carton_or_demo_vehicle")
    assert record["frame_index"] == 4
    assert record["detections"] == [{"class_id": 0, "class_name": "carton_or_demo_vehicle", "confidence": 0.87, "bbox_xyxy": [1, 2, 11, 22], "width": 10.0, "height": 20.0, "area": 200.0}]
