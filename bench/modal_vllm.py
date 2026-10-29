"""A big open-weights model behind vLLM on Modal: an OpenAI-compatible endpoint for bench/swe_api.py.

  download   CPU: the preset's weights into the cortexnet-weights Volume (once per model)
  serve      one container running `vllm serve` (prefix caching on, the preset's tool-call parser); at most one
             container, so spend is its GPUs' time; it scales to zero after SCALEDOWN seconds without requests

Presets (CORTEXNET_VLLM_PRESET, read when the app is run or deployed); every one is an FP8 build that is not gated on
Hugging Face, so no token leaves the machine:
  llama-3.3-70b     Llama 3.3 70B Instruct, one H200
  llama-4-scout     Llama 4 Scout 17B-16E (109B MoE), one H200
  llama-4-maverick  Llama 4 Maverick 17B-128E (400B MoE, 417 GB), four H200 with tensor parallelism
Llama 4 uses vLLM's pythonic tool-call parser with vLLM's own Llama 4 template (its docs' recommendation).

The endpoint is public, so vLLM requires an API key: a random key generated locally (CORTEXNET_VLLM_KEY, handed to the
container as a Modal secret), never a personal credential.

  CORTEXNET_VLLM_PRESET=llama-4-maverick CORTEXNET_VLLM_KEY=<random> modal run bench/modal_vllm.py::download
  CORTEXNET_VLLM_PRESET=llama-4-maverick CORTEXNET_VLLM_KEY=<random> modal deploy bench/modal_vllm.py
  python -I bench/swe_api.py run --openai <url>/v1 --model llama-4-maverick --key-env CORTEXNET_VLLM_KEY \\
      --protocol v2
  modal app stop --yes cortexnet-vllm                                   # stop paying at once
"""
import os

import modal

VLLM = "0.32.0"
LLAMA4_TEMPLATE = (f"https://raw.githubusercontent.com/vllm-project/vllm/v{VLLM}/examples/"
                   "tool_chat_template_llama4_pythonic.jinja")
PRESETS = {
    "llama-3.3-70b": dict(model="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic", gpu="H200", tp=1,
                          parser="llama3_json", template=None, max_len=65536),
    "llama-4-scout": dict(model="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic", gpu="H200", tp=1,
                          parser="llama4_pythonic", template="/root/llama4_pythonic.jinja", max_len=65536),
    "llama-4-maverick": dict(model="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8", gpu="H200:4", tp=4,
                             parser="llama4_pythonic", template="/root/llama4_pythonic.jinja", max_len=131072),
}
NAME = os.environ.get("CORTEXNET_VLLM_PRESET", "llama-4-maverick")
P = PRESETS[NAME]
DIR = "/w/" + P["model"].replace("/", "__")
SCALEDOWN = 120

app = modal.App("cortexnet-vllm")
weights = modal.Volume.from_name("cortexnet-weights", create_if_missing=True)
compile_cache = modal.Volume.from_name("cortexnet-vllm-cache", create_if_missing=True)
dl_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=0.30", "hf_xet")
# CUDA 13.0 developer image (torch 2.13 is built for CUDA 13.0): Llama 4's MoE kernels and FlashInfer JIT-compile with
# nvcc at start-up, and a slim image without the toolkit failed there after loading the weights (70B: the sampler;
# Maverick: the memory profile run). The sampler stays on vLLM's PyTorch path to skip one compile at start-up.
image = (modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
         .pip_install(f"vllm=={VLLM}")
         .run_commands(f"python -c \"import urllib.request; urllib.request.urlretrieve('{LLAMA4_TEMPLATE}', "
                       "'/root/llama4_pythonic.jinja')\"")
         .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "CORTEXNET_VLLM_PRESET": NAME}))


@app.function(image=dl_image, volumes={"/w": weights}, cpu=8, memory=16384, timeout=4 * 3600)
def download():
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(DIR, "config.json")):
        snapshot_download(P["model"], local_dir=DIR, ignore_patterns=["*.pth", "original/*"])
        weights.commit()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DIR) for f in fs)
    return f"{P['model']}: {size / 1e9:.1f} GB in {DIR}"


@app.function(image=dl_image, volumes={"/w": weights}, cpu=8, memory=32768, timeout=1800)
def read_speed(n_files: int = 8, threads: int = 8, block_mb: int = 64):
    """Sequential read throughput of the preset's shards from the Volume (a CPU container: cents, not GPU dollars).
    The first Maverick start read shards at random through the Volume's 9P mount: 2 of 84 in 5 minutes."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    files = sorted(os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(".safetensors"))[:n_files]

    def read(p):
        n = 0
        with open(p, "rb", buffering=0) as f:
            while chunk := f.read(block_mb << 20):
                n += len(chunk)
        return n
    t = time.time()
    with ThreadPoolExecutor(threads) as ex:
        total = sum(ex.map(read, files))
    dt = time.time() - t
    return f"{len(files)} shards, {total / 1e9:.1f} GB in {dt:.0f} s: {total / 1e6 / dt:.0f} MB/s"


@app.function(image=image, gpu=P["gpu"], volumes={"/w": weights, "/root/.cache/vllm": compile_cache},
              secrets=[modal.Secret.from_local_environ(["CORTEXNET_VLLM_KEY"])], scaledown_window=SCALEDOWN,
              timeout=4 * 3600, max_containers=1)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=2400)
def serve():
    import subprocess
    cmd = ["vllm", "serve", DIR, "--served-model-name", NAME, "--host", "0.0.0.0", "--port", "8000",
           "--api-key", os.environ["CORTEXNET_VLLM_KEY"], "--max-model-len", str(P["max_len"]),
           "--tensor-parallel-size", str(P["tp"]), "--kv-cache-dtype", "fp8", "--enable-prefix-caching",
           "--enable-prompt-tokens-details", "--gpu-memory-utilization", "0.92",
           "--enable-auto-tool-choice", "--tool-call-parser", P["parser"],
           # the Volume's 9P mount is not detected as a network FS, so vLLM reads shards at random (Maverick: 2 of 84
           # in 5 minutes); prefetching reads each file sequentially into page cache (read_speed: ~880 MB/s)
           "--safetensors-load-strategy", "prefetch"]
    if P["template"]:
        cmd += ["--chat-template", P["template"]]
    subprocess.Popen(cmd)


@app.local_entrypoint()
def speed(n_files: int = 8, threads: int = 8):
    print(read_speed.remote(n_files, threads))


@app.function(image=image, cpu=2, memory=8192, timeout=600)
def versions():
    import subprocess

    import torch
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return f"torch {torch.__version__}, built for CUDA {torch.version.cuda}; nvcc: {nvcc}"


@app.local_entrypoint()
def check():
    print(versions.remote())
