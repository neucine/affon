"""Offline test oracle: Transformers 4.56.2, Pillow bilinear RGB8 preprocessing."""
import json
from pathlib import Path
import numpy as np
from transformers import MobileNetV2ImageProcessor
processor = MobileNetV2ImageProcessor(size={'shortest_edge': 8}, crop_size={'height': 6, 'width': 6})
cases = []
for height, width in [(11, 11), (11, 15), (15, 11), (3, 5)]:
    y, x, c = np.indices((height, width, 3))
    rgb = ((x * 3 + y * 5 + c * 47) % 256).astype(np.uint8)
    cases.append({'height': height, 'width': width, 'pixels': processor(images=rgb, input_data_format='channels_last', return_tensors='np')['pixel_values'].tolist()})
Path(__file__).with_name('fixtures').joinpath('mobilenet-processor.json').write_text(json.dumps(cases))
