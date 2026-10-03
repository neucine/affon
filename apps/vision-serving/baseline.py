"""Matching V2 HTTP subset using Pillow/NumPy preprocessing and ONNX Runtime."""
import json
import os
from pathlib import Path
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

import numpy as np
from PIL import Image
import onnxruntime as ort

ort.disable_telemetry_events()
ROOT = Path(os.environ["MODEL_DIR"])
NAME = os.environ.get("MODEL_NAME", "vision")
if not re.fullmatch(r"[a-zA-Z0-9_-]+", NAME):
    raise ValueError("Invalid MODEL_NAME")
BASE = f"/v2/models/{NAME}"
options = ort.SessionOptions()
options.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
options.inter_op_num_threads = 1
session = ort.InferenceSession(str(ROOT / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
if len(session.get_inputs()) != 1 or session.get_inputs()[0].name != "pixels" or session.get_inputs()[0].shape != [1, 3, 224, 224]:
    raise ValueError("Expected pixels [1,3,224,224] input")
if len(session.get_outputs()) != 1 or session.get_outputs()[0].name != "logits":
    raise ValueError("Expected logits output")


def preprocess(image):
    config = json.loads((ROOT / "source/preprocessor_config.json").read_text())
    if config.get("image_processor_type") != "MobileNetV2ImageProcessor":
        raise ValueError("This service requires MobileNetV2 preprocessing")
    if not all(config.get(k) is True for k in ["do_resize", "do_center_crop", "do_rescale", "do_normalize"]) or config["resample"] != 2:
        raise ValueError("Unsupported processor configuration")
    height, width = image.shape[:2]
    short = config["size"]["shortest_edge"]
    rh = short if height <= width else short * height // width
    rw = short if width <= height else short * width // height
    resized = np.asarray(Image.fromarray(image).resize((rw, rh), Image.Resampling.BILINEAR))
    ch, cw = config["crop_size"]["height"], config["crop_size"]["width"]
    if ch > rh or cw > rw:
        raise ValueError("Crop must fit resized image")
    top, left = (rh - ch) // 2, (rw - cw) // 2
    # Affon rounds after the double-precision rescale product, then subtract/divide.
    data = (resized[top:top+ch, left:left+cw].astype(np.float64) * config["rescale_factor"]).astype(np.float32)
    data = ((data - np.array(config["image_mean"], dtype=np.float32)) / np.array(config["image_std"], dtype=np.float32))
    return np.ascontiguousarray(data.transpose(2, 0, 1)[None])


probe = session.run(None, {"pixels": preprocess(np.zeros((1, 1, 3), dtype=np.uint8))})[0]
if probe.ndim != 2 or probe.shape[0] != 1:
    raise ValueError("Expected batch-one classification logits")


def validate(value):
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    if "id" in value and not isinstance(value["id"], str):
        raise ValueError("id must be a string")
    if value.get("parameters"):
        raise ValueError("Request parameters are unsupported")
    if "outputs" in value and value["outputs"] != [{"name": "logits"}]:
        raise ValueError("Only logits output is supported")
    inputs = value.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("Expected one rgb input")
    rgb = inputs[0]
    if not isinstance(rgb, dict) or rgb.get("name") != "rgb" or rgb.get("datatype") != "UINT8" or "parameters" in rgb:
        raise ValueError("Expected rgb UINT8 input without parameters")
    shape = rgb.get("shape")
    if not isinstance(shape, list) or len(shape) != 4 or any(type(n) is not int for n in shape) or shape[0] != 1 or shape[3] != 3 or not all(1 <= n <= 512 for n in shape[1:3]) or max(shape[1:3]) > 4 * min(shape[1:3]):
        raise ValueError("Expected shape [1,height,width,3], height/width 1–512")
    data = rgb.get("data")
    if not isinstance(data, list) or len(data) != shape[1] * shape[2] * 3 or any(type(v) is not int or not 0 <= v <= 255 for v in data):
        raise ValueError("Expected flat RGB8 data matching shape")
    return shape, data


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def reply(self, value=None, status=200):
        body = b"" if value is None else json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ["/v2/health/live", "/v2/health/ready", BASE + "/ready"]:
            return self.reply()
        if path == "/v2":
            return self.reply({"name": "onnxruntime-vision", "version": ort.__version__, "extensions": []})
        if path == BASE:
            return self.reply({"name": NAME, "platform": "onnxruntime", "inputs": [{"name": "rgb", "datatype": "UINT8", "shape": [1, -1, -1, 3]}],
                               "outputs": [{"name": "logits", "datatype": "FP32", "shape": list(probe.shape)}]})
        return self.reply({"error": "Use POST" if path == BASE + "/infer" else "Not found"}, 405 if path == BASE + "/infer" else 404)

    def do_POST(self):
        # Close rejected connections so unread bodies cannot be parsed as requests.
        self.close_connection = True
        if self.path.split("?")[0] != BASE + "/infer":
            return self.reply({"error": "Not found"}, 404)
        if self.headers.get_content_type() != "application/json":
            return self.reply({"error": "Use application/json"}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4 * 1024 * 1024:
                return self.reply({"error": "Body too large"}, 413)
            if length <= 0:
                raise ValueError("Expected request body")
            value = json.loads(self.rfile.read(length))
            shape, data = validate(value)
        except (ValueError, TypeError):
            return self.reply({"error": "Invalid inference request"}, 400)
        try:
            start = time.perf_counter()
            pixels = preprocess(np.asarray(data, dtype=np.uint8).reshape(shape[1:]))
            prepared = time.perf_counter()
            logits = session.run(None, {"pixels": pixels})[0]
            output = logits.reshape(-1).tolist()
            if not np.isfinite(logits).all():
                raise ValueError("Nonfinite model output")
            finished = time.perf_counter()
            return self.reply({"model_name": NAME, **({"id": value["id"]} if "id" in value else {}),
                "parameters": {"preprocess_ms": (prepared-start)*1000, "inference_ms": (finished-prepared)*1000},
                "outputs": [{"name": "logits", "datatype": "FP32", "shape": list(logits.shape), "data": output}]})
        except Exception as error:
            return self.reply({"error": str(error)}, 500)


if __name__ == "__main__":
    print(json.dumps({"ready": True, "engine": "onnxruntime"}), flush=True)
    HTTPServer((os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8080"))), Handler).serve_forever()
