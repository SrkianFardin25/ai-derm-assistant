import sys
import os
import time
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.api.main import app
from src.infrastructure.onnx_engine import ONNXInferenceEngine

def run_benchmark():
    engine = ONNXInferenceEngine()
    img = Image.new("RGB", (380, 380), color=(180, 110, 80))

    # Engine warm-up
    for _ in range(3):
        engine.run_inference(img)

    t_engine = []
    for _ in range(15):
        t0 = time.perf_counter()
        logits, cam = engine.run_inference(img)
        t_engine.append((time.perf_counter() - t0) * 1000)

    # API End-to-end benchmark
    client = TestClient(app)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    # API warm-up
    for _ in range(2):
        client.post(
            "/predict?generate_summary=false&save_to_cloud=false",
            files={"file": ("lesion.jpg", io.BytesIO(buf.getvalue()), "image/jpeg")}
        )

    t_e2e = []
    last_response = None
    for _ in range(15):
        t0 = time.perf_counter()
        last_response = client.post(
            "/predict?generate_summary=false&save_to_cloud=false",
            files={"file": ("lesion.jpg", io.BytesIO(buf.getvalue()), "image/jpeg")}
        )
        t_e2e.append((time.perf_counter() - t0) * 1000)

    d = last_response.json()
    print("=== BENCHMARK RESULTS ===")
    print(f"ONNX CAM Engine Latency (ms): Mean={np.mean(t_engine):.2f}, Median={np.median(t_engine):.2f}, Min={np.min(t_engine):.2f}, P95={np.percentile(t_engine, 95):.2f}")
    print(f"End-to-End API /predict Latency (ms): Mean={np.mean(t_e2e):.2f}, Median={np.median(t_e2e):.2f}, Min={np.min(t_e2e):.2f}, P95={np.percentile(t_e2e, 95):.2f}")
    print(f"CAM Resolution: {cam.shape[0]}x{cam.shape[1]} ({cam.shape[0]*cam.shape[1]} points)")
    print(f"all_probs Sum: {sum(d['all_probs']):.8f}")
    print(f"HTTP Status: {last_response.status_code}")
    print(f"is_calibrated: {d.get('is_calibrated')}")
    print(f"explanation_method: {d.get('explanation_method')}")
    print(f"temperature: {d.get('temperature')}")

if __name__ == "__main__":
    run_benchmark()
