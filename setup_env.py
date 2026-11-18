"""Create a local venv, install dependencies into it, and optionally pre-download models."""
import argparse
import os
import subprocess
import venv
from pathlib import Path

PACKAGES = ["transformers>=4.51,<5", "accelerate", "numpy", "huggingface_hub", "safetensors"]

CHECK = (
    "import torch, transformers, numpy, accelerate;"
    "print('torch', torch.__version__, '| transformers', transformers.__version__,"
    " '| numpy', numpy.__version__, '| accelerate', accelerate.__version__);"
    "print('cuda available:', torch.cuda.is_available());"
    "print('gpu:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None"
)

DOWNLOAD = (
    "import sys;from huggingface_hub import snapshot_download;"
    "[snapshot_download(m, allow_patterns=['*.json','*.txt','*.safetensors']) for m in sys.argv[1:]]"
)


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=".venv")
    ap.add_argument("--cuda", default="cu124", help="cu121, cu124, cu126 or cpu")
    ap.add_argument("--download", action="store_true", help="pre-download model weights")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-32B", "Qwen/Qwen3-0.6B"])
    a = ap.parse_args()

    env = Path(a.venv)
    if not env.exists():
        venv.create(env, with_pip=True)
    py = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "torch", "--index-url", f"https://download.pytorch.org/whl/{a.cuda}"])
    run([py, "-m", "pip", "install", *PACKAGES])
    run([py, "-c", CHECK])

    if a.download:
        run([py, "-c", DOWNLOAD, *a.models])

    print(f"\nDone. Activate with: source {env}/bin/activate" if os.name != "nt"
          else f"\nDone. Activate with: {env}\\Scripts\\Activate.ps1")


if __name__ == "__main__":
    main()
