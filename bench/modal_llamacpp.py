"""The laptop's model and server, at scale: Llama 3.1 8B Instruct Q4_K_M in llama-server on a Modal L40S (or Meta's
small Llama 3.2 3B / 1B on an L4: CORTEXNET_LLAMA_MODEL=3b or 1b).

Same GGUF (bartowski/Meta-Llama-3.1-8B-Instruct-GGUF, Q4_K_M) and the same server (llama.cpp's llama-server with
--jinja, all layers on the GPU) as the 6 GB laptop runs, so the held-out sessions (bench/heldout.py) change the scale
and the questions, not the model. One deployment per context size; each has SLOTS parallel slots of exactly that
context (llama-server splits -c across -np), so a request over the window fails the same way it did on the laptop.

  modal run bench/modal_llamacpp.py::download                         # the GGUF into a Volume, once
  CORTEXNET_LLAMA_CTX=8192 CORTEXNET_LLAMA_KEY=<random> modal deploy bench/modal_llamacpp.py
  python -I bench/heldout.py run --openai <url>/v1 --ctx 8192 ...
  modal app stop --yes cortexnet-llama-8k                              # stop paying at once

The endpoint is public, so llama-server requires an API key: a random key generated locally (CORTEXNET_LLAMA_KEY,
handed to the container as a Modal secret), never a personal credential.
"""
import os

import modal

# CORTEXNET_LLAMA_MODEL: the laptop's 8B (default), or Meta's small Llama 3.2 models for the small-model workflow
# runs; all Q4_K_M from bartowski (not gated). The small ones run on an L4 (also sm_89; an L40S when Modal has no L4
# free) with a 128k-token KV pool.
MODELS = {"8b": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "L40S", 262144),
          "3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf", ["L4", "L40S"], 131072),
          "1b": ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", ["L4", "L40S"], 131072)}
MODEL = os.environ.get("CORTEXNET_LLAMA_MODEL", "8b")
REPO, FILE, GPU, POOL = MODELS[MODEL]
CTX = int(os.environ.get("CORTEXNET_LLAMA_CTX", "8192"))
SLOTS = max(1, POOL // CTX)        # the KV pool (256k tokens = 32 GB at f16 for the 8B) split into slots of exactly CTX
SCALEDOWN = 180
# b11936 = the build the laptop runs used; later builds' PEG tool-call parser rejects Llama 3.1 8B's calls (HTTP 500
# "does not match the expected peg-native format" on the first turn)
BUILD = os.environ.get("CORTEXNET_LLAMA_BUILD", "b11936")

app = modal.App(f"cortexnet-llama-{CTX // 1024}k" + ("" if MODEL == "8b" else f"-{MODEL}")
                + ("" if BUILD == "b11936" else f"-{BUILD}"))
weights = modal.Volume.from_name("cortexnet-gguf", create_if_missing=True)
dl_image = (modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=0.30", "hf_xet")
            .env({"CORTEXNET_LLAMA_MODEL": MODEL}))     # the container re-imports this file: same model
# llama.cpp publishes server images for only some builds (none for b11936), so the server is compiled from the tag:
# CUDA for the L40S only (sm_89), the llama-server target only
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
         .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev", "libgomp1")
         .run_commands(f"git clone --depth 1 --branch {BUILD} https://github.com/ggml-org/llama.cpp /src",
                       "cmake -S /src -B /src/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
                       "-DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined",
                       "cmake --build /src/build --target llama-server -j 16", "cp /src/build/bin/llama-server /usr/local/bin/",
                       "cp /src/build/bin/*.so* /usr/local/lib/ 2>/dev/null; ldconfig", gpu=None)
         .env({"CORTEXNET_LLAMA_CTX": str(CTX), "CORTEXNET_LLAMA_BUILD": BUILD, "CORTEXNET_LLAMA_MODEL": MODEL}))


@app.function(image=dl_image, volumes={"/w": weights}, cpu=4, memory=8192, timeout=3600)
def download():
    from huggingface_hub import hf_hub_download
    p = os.path.join("/w", FILE)
    if not os.path.exists(p):
        hf_hub_download(REPO, FILE, local_dir="/w")
        weights.commit()
    return f"{FILE}: {os.path.getsize(p) / 1e9:.2f} GB"


@app.function(image=image, gpu=GPU, cpu=8, memory=32768, volumes={"/w": weights},
              secrets=[modal.Secret.from_local_environ(["CORTEXNET_LLAMA_KEY"])], scaledown_window=SCALEDOWN,
              timeout=6 * 3600, max_containers=1)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8080, startup_timeout=600)
def serve():
    import subprocess
    subprocess.Popen(["/usr/local/bin/llama-server", "--host", "0.0.0.0", "--port", "8080",
                      "-m", os.path.join("/w", FILE), "--jinja", "-ngl", "99", "-c", str(CTX * SLOTS), "-np", str(SLOTS), "--alias", f"llama-{MODEL}",
                      "--api-key", os.environ["CORTEXNET_LLAMA_KEY"], "--metrics"])
