#!/usr/bin/env python3

import argparse
import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pynvml
import requests
from flask import Flask, Response, jsonify, request

#Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("telemetry_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


#Polls GPU metrics at a set frequency and stores readings in a buffer
class GPUSampler:

    def __init__(self, sample_hz: int = 10):
        #Initialize NVML and set up sampling state
        pynvml.nvmlInit()
        self.handle    = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.sample_hz = sample_hz
        self.buffer: list[dict] = []
        self._running  = False
        self._lock     = threading.Lock()

    def start(self):
        #Start the background sampling thread
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"GPU sampler started at {self.sample_hz}Hz")

    def stop(self):
        #Stop the sampling thread
        self._running = False
        logger.info("GPU sampler stopped")

    def _run(self):
        #Continuously poll GPU power, utilization, and memory at set interval
        while self._running:
            ts = time.time()
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                util     = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem      = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                with self._lock:
                    self.buffer.append({
                        "ts":           ts,
                        "power_w":      power_mw / 1000.0,
                        "gpu_util_pct": util.gpu,
                        "mem_used_gb":  mem.used / (1024 ** 3),
                    })

            except pynvml.NVMLError as e:
                logger.warning(f"GPU sample error: {e}")

            time.sleep(1.0 / self.sample_hz)

    def get_window(self, t_start: float, t_end: float) -> list[dict]:
        #Return samples within a given time window
        with self._lock:
            return [s for s in self.buffer if t_start <= s["ts"] <= t_end]

    def get_all(self) -> list[dict]:
        #Return all buffered samples
        with self._lock:
            return list(self.buffer)

    def shutdown(self):
        #Stop sampling and shut down NVML
        self.stop()
        pynvml.nvmlShutdown()


#Thread-safe store for per-request records
class RequestStore:

    def __init__(self):
        self._records: list[dict] = []
        self._lock = threading.Lock()

    def add(self, record: dict):
        #Add a new request record
        with self._lock:
            self._records.append(record)

    def get_all(self) -> list[dict]:
        #Return all stored records
        with self._lock:
            return list(self._records)

    def count(self) -> int:
        #Return total number of records
        with self._lock:
            return len(self._records)


#Join request records with GPU samples and compute aggregate features per request
def build_dataset(request_store: RequestStore,
                  gpu_sampler:   GPUSampler) -> pd.DataFrame:
    rows    = []
    missing = 0

    for req in request_store.get_all():
        #Get GPU samples that overlap with this request's time window
        window = gpu_sampler.get_window(req["t_start"], req["t_end"])

        if not window:
            logger.warning(
                f"No GPU samples for req {req['request_id'][:8]} "
                f"(latency={req['latency_s']}s) — increase --sample-hz "
                f"if many short requests are missing"
            )
            missing += 1
            continue

        power_series = [s["power_w"]      for s in window]
        util_series  = [s["gpu_util_pct"]  for s in window]
        mem_series   = [s["mem_used_gb"]   for s in window]

        tokens = req.get("tokens_out") or 1
        energy = float(np.trapz(power_series))

        #Build one row of features per request
        rows.append({
            "request_id":       req["request_id"],
            "model":            req["model"],
            "t_start":          req["t_start"],
            "t_end":            req["t_end"],
            "latency_s":        req["latency_s"],
            "tokens_out":       req.get("tokens_out"),
            "tokens_in":        req.get("tokens_in"),
            "prompt_hash":      req.get("prompt_hash"),
            "prompt_preview":   req.get("prompt_preview"),
            "endpoint":         req.get("endpoint"),
            "status_code":      req.get("status_code"),
            "power_mean_w":     round(float(np.mean(power_series)), 3),
            "power_max_w":      round(float(np.max(power_series)),  3),
            "power_min_w":      round(float(np.min(power_series)),  3),
            "power_std_w":      round(float(np.std(power_series)),  3),
            "power_auc":        round(energy, 3),
            "power_per_token":  round(energy / tokens, 5),
            "gpu_util_mean":    round(float(np.mean(util_series)), 3),
            "gpu_util_max":     round(float(np.max(util_series)),  3),
            "gpu_util_std":     round(float(np.std(util_series)),  3),
            "mem_used_mean_gb": round(float(np.mean(mem_series)),  3),
            "mem_used_max_gb":  round(float(np.max(mem_series)),   3),
            "n_samples":        len(window),
        })

    if missing:
        logger.warning(
            f"{missing}/{request_store.count()} requests had no GPU samples."
        )

    return pd.DataFrame(rows)


#Save telemetry CSV, request JSON log, and raw GPU samples to disk
def save_results(request_store: RequestStore,
                 gpu_sampler:   GPUSampler,
                 model_name:    str,
                 output_dir:    str = "./output",
                 label:         int = None) -> pd.DataFrame:

    model_short = model_name.split("/")[-1]
    out_path    = Path(output_dir) / model_short
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = build_dataset(request_store, gpu_sampler)

    if label is not None:
        df.insert(0, "label", label)

    csv_path = out_path / f"telemetry_{ts}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Telemetry dataset → {csv_path}  ({len(df)} rows, label={label})")

    req_path = out_path / f"requests_{ts}.json"
    with open(req_path, "w") as f:
        json.dump(request_store.get_all(), f, indent=2, default=str)
    logger.info(f"Request log       → {req_path}")

    raw_path = out_path / f"raw_gpu_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(gpu_sampler.get_all(), f, indent=2)
    logger.info(f"Raw GPU telemetry → {raw_path}")

    return df


#Create Flask proxy app that intercepts requests between HarmBench and vLLM
def create_proxy(model_name:    str,
                 vllm_url:      str,
                 request_store: RequestStore,
                 gpu_sampler:   GPUSampler) -> Flask:
    app = Flask(__name__)

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    def _extract_prompt_meta(body: dict, endpoint: str) -> tuple[str, str]:
        #Hash and preview the prompt without storing full text
        if "chat" in endpoint:
            messages = body.get("messages", [])
            text = " ".join(m.get("content", "") for m in messages)
        else:
            text = body.get("prompt", "")

        return (
            hashlib.sha256(text.encode()).hexdigest()[:16],
            text[:120].replace("\n", " "),
        )

    def _proxy(endpoint: str) -> Response:
        #Forward request to vLLM, record timing and metadata
        request_id              = str(uuid.uuid4())
        body                    = request.get_json(force=True, silent=True) or {}
        prompt_hash, preview    = _extract_prompt_meta(body, endpoint)

        t_start = time.time()
        try:
            upstream = requests.post(
                f"{vllm_url}{endpoint}",
                json={**body, "user": request_id},
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            upstream.raise_for_status()
            result = upstream.json()
            status = upstream.status_code
            error  = None

        except Exception as e:
            logger.error(f"Upstream error [{request_id[:8]}]: {e}")
            result = {}
            status = 502
            error  = str(e)

        t_end = time.time()

        usage = result.get("usage", {})
        request_store.add({
            "request_id":     request_id,
            "model":          model_name,
            "endpoint":       endpoint,
            "t_start":        t_start,
            "t_end":          t_end,
            "latency_s":      round(t_end - t_start, 4),
            "tokens_in":      usage.get("prompt_tokens"),
            "tokens_out":     usage.get("completion_tokens"),
            "prompt_hash":    prompt_hash,
            "prompt_preview": preview,
            "status_code":    status,
            "error":          error,
        })

        logger.info(
            f"req={request_id[:8]} | "
            f"latency={round(t_end - t_start, 2)}s | "
            f"tok_in={usage.get('prompt_tokens')} "
            f"tok_out={usage.get('completion_tokens')} | "
            f"total={request_store.count()}"
        )

        return Response(
            json.dumps(result),
            status=status,
            mimetype="application/json"
        )

    #Route chat and completion requests through the proxy
    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        return _proxy("/v1/chat/completions")

    @app.route("/v1/completions", methods=["POST"])
    def completions():
        return _proxy("/v1/completions")

    #Pass-through routes for model listing and health checks
    @app.route("/v1/models", methods=["GET"])
    def models():
        r = requests.get(f"{vllm_url}/v1/models", timeout=5)
        return Response(r.text, status=r.status_code, mimetype="application/json")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "requests_seen": request_store.count()})

    #Collector control endpoints for status and manual save
    @app.route("/collector/status", methods=["GET"])
    def collector_status():
        #Return current collection counts
        return jsonify({
            "requests_collected": request_store.count(),
            "gpu_samples":        len(gpu_sampler.get_all()),
            "model":              model_name,
        })

    @app.route("/collector/save", methods=["POST"])
    def collector_save():
        #Trigger save and return row/column summary
        label = request.args.get("label", default=None, type=int)
        df = save_results(request_store, gpu_sampler, model_name, label=label)
        return jsonify({
            "rows_saved": len(df),
            "columns":    list(df.columns),
        })

    return app


#Parse args, start GPU sampler, launch proxy server, and save on exit
def main():
    parser = argparse.ArgumentParser(
        description="Passive telemetry proxy — sits between HarmBench and vLLM"
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B",
        help="Exact model name loaded in vLLM (used for output folder naming)"
    )
    parser.add_argument(
        "--vllm-port", default=8000, type=int,
        help="Port vLLM is listening on (default 8000)"
    )
    parser.add_argument(
        "--proxy-port", default=8001, type=int,
        help="Port this proxy listens on — point HarmBench here (default 8001)"
    )
    parser.add_argument(
        "--sample-hz", default=10, type=int,
        help="GPU polling frequency in Hz (default 10 = every 100ms)"
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="Root directory for output files (default ./output)"
    )
    parser.add_argument(
        "--label", default=None, type=int, choices=[0, 1],
        help="Label to stamp on every request in this run: 1=malicious, 0=benign"
    )
    args = parser.parse_args()

    vllm_url = f"http://localhost:{args.vllm_port}"

    logger.info("=" * 55)
    logger.info(f"  Model      : {args.model}")
    logger.info(f"  vLLM       : {vllm_url}")
    logger.info(f"  Proxy      : http://localhost:{args.proxy_port}  ← point HarmBench here")
    logger.info(f"  GPU sample : {args.sample_hz}Hz")
    logger.info(f"  Label      : {args.label}  (1=malicious, 0=benign, None=unlabeled)")
    logger.info(f"  Output     : {args.output_dir}/{args.model.split('/')[-1]}/")
    logger.info("=" * 55)

    gpu_sampler   = GPUSampler(sample_hz=args.sample_hz)
    request_store = RequestStore()
    gpu_sampler.start()

    app = create_proxy(
        model_name=args.model,
        vllm_url=vllm_url,
        request_store=request_store,
        gpu_sampler=gpu_sampler,
    )

    try:
        app.run(host="0.0.0.0", port=args.proxy_port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Interrupted — saving results...")
    finally:
        save_results(request_store, gpu_sampler, args.model, args.output_dir, args.label)
        gpu_sampler.shutdown()


if __name__ == "__main__":
    main()
