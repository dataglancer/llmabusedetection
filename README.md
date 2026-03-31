# Project File Descriptions
---


These files were used for a reported named "LLM Abuse Detection via
GPU Power Telemetry" where the goal was to determine if power telemetry could be used as a suitable indicator for abuse detection. 
Note, the telemetry logs generated with telemetry collector proxy and analyzed in the notebooks are note included.

## 1. `model_telemetry_collector_ver4.py`

A passive GPU telemetry proxy designed for jailbreak detection research. It sits between HarmBench (a prompt evaluation framework) and a vLLM inference server, intercepting every request to record GPU power, utilization, and memory usage during inference. For each request, it time-stamps the start and end, polls GPU metrics in the background at a configurable frequency (default 10Hz), and joins the two data streams to produce per-request aggregate features (e.g., mean power, energy per token). Results are saved as a CSV for ML use, a JSON request log, and a raw GPU sample dump. The script is run as a Flask proxy on port 8001 while vLLM runs on port 8000; HarmBench is pointed at the proxy port. An optional label flag (0=benign, 1=malicious) can be stamped on all requests in a run for supervised learning workflows.

---

## 2. `WildbreakGeneration.ipynb`

A Jupyter notebook that loads the WildJailbreak dataset (from HuggingFace, `allenai/wildjailbreak`) in streaming mode and sends prompts to a vLLM inference endpoint to collect model responses. It handles both harmful and benign subsets of the dataset. Harmful prompts use the adversarial version if available, otherwise falling back to the vanilla prompt. Each request is logged with a unique session ID, the model's response, latency, and an abuse label (1 for harmful, 0 for benign). A small pilot run of 25 items is included for testing before larger full-dataset runs (up to 8000 items). Results are saved as CSV and JSONL files for downstream analysis.

---

## 3. `JailbreakingNB.ipynb`

A Jupyter notebook that loads the JailbreakBench behaviors dataset (`JailbreakBench/JBB-Behaviors`) and sends each harmful goal prompt to a vLLM inference server, recording the model response, latency, and category. It also loads the `lmsys/lmsys-chat-1m` dataset to extract and filter 1000 clean benign English prompts (non-flagged, reasonable length), which are then sent to the same vLLM endpoint to generate benign baseline traffic. Results from both runs are saved as JSONL files and analyzed by category for latency and token statistics. The notebook also includes connectivity checks, model listing, and a prompt length distribution plot for the benign set.

---

## 4. `abuse_detection_eda_ml.ipynb`

A large Jupyter notebook covering the full EDA and ML pipeline for LLM abuse detection using GPU power telemetry. It loads benign and abuse telemetry CSVs for three models (Llama, Phi, Qwen), combines them into a master DataFrame, and performs extensive exploratory analysis including KDE plots of power features, correlation heatmaps, boxplots, token throughput analysis, and Mann-Whitney U significance tests. The ML section trains four classifiers (Random Forest, XGBoost, LightGBM, and a PyTorch MLP) with stratified train/test splits, 5-fold cross-validation, and class imbalance handling. It evaluates all models via confusion matrices, ROC curves, precision-recall curves, and feature importance charts. It also includes diagnostic cells for detecting proxy label leakage, a cross-model holdout generalization test, power/energy cost analysis, and out-of-distribution evaluation on a separate `qwen_wild` dataset collected from WildJailbreak prompts.

---

## 5. `time_branch_abuse_detection_eda_ml.ipynb`

Identical to `abuse_detection_eda_ml.ipynb` in every way except one: in Cell 13 (the feature set definition cell), `ACTIVE_FEATURE_SET` is set to `'time_only'` instead of `'telemetry_only'`. This means the ML classifiers in this branch are trained and evaluated using only timing-based features (`latency_s`, `tokens_out`, `tokens_in`) rather than GPU power telemetry, serving as a timing-only ablation baseline for comparison against the full telemetry model.

---
