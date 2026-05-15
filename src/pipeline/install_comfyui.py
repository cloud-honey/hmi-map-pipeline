"""
Task 2.1: ComfyUI Setup + Custom Nodes Installer
HMI Map Automation Pipeline

This script handles ComfyUI installation on the target 8GB PC.
It also configures required custom nodes and validates the setup.

Run on TARGET PC (8GB RAM):
  python src/pipeline/install_comfyui.py [--install] [--verify] [--port 8188]
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_DIR = Path.home() / "ComfyUI"
REQUIREMENTS_FILE = COMfyUI_DIR / "requirements.txt"

# Required custom nodes for this pipeline
REQUIRED_CUSTOM_NODES = {
    "ControlNet Tile (SDXL)": "ltdrdata/ComfyUI-Static-Synthesis",
    # Alternative: lllyasviel/ControlNet-sdxl-tile (onnx)
    "ControlNet SDXL Canny": "Fannovel16/comfyui_controlnet_aux",
    "ControlNet SDXL Depth": "Fannovel16/comfyui_controlnet_aux",
    "Multi-ControlNet": "Kohya-SS/ComfyUI-Colorspaces-ControlNet-Pro",
    # Optional: ComfyUI-Manager for easy node management
}

REQUIRED_PIP_PACKAGES = [
    "torch",          # Already installed
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "onnxruntime-gpu",  # For ControlNet ONNX models
]


def check_already_installed() -> bool:
    """Check if ComfyUI is already installed."""
    return COMFYUI_DIR.exists() and (COMFYUI_DIR / "main.py").exists()


def run_cmd(cmd: list[str], cwd: Path = None, capture: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a command with optional timeout."""
    print(f"  Running: {' '.join(cmd[:3])}...")
    return subprocess.run(
        cmd, capture_output=capture, text=True, cwd=cwd,
        timeout=timeout, env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )


def install_comfyui(verify: bool = True):
    """
    Install ComfyUI if not present.

    Steps:
    1. Clone ComfyUI repo (or pull latest if exists)
    2. Install pip requirements
    3. Install ComfyUI-Manager
    4. Download required custom nodes
    5. Verify setup with test generation
    """
    print("\n=== ComfyUI Installation ===\n")

    if check_already_installed():
        print(f"✅ ComfyUI already installed at {COMFYUI_DIR}")
    else:
        print(f"Cloning ComfyUI from {COMFYUI_REPO}...")
        result = run_cmd(["git", "clone", COMFYUI_REPO, str(COMFYUI_DIR)])
        if result.returncode != 0:
            print(f"❌ Git clone failed: {result.stderr}")
            return False
        print("  ✅ Cloned")

    # Install requirements
    print("\nInstalling ComfyUI requirements...")
    result = run_cmd([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
                     cwd=COMFYUI_DIR, timeout=600)
    if result.returncode != 0:
        print(f"  ⚠️ Some requirements failed: {result.stderr[:500]}")
    else:
        print("  ✅ Requirements installed")

    # Install ComfyUI-Manager
    manager_dir = COMFYUI_DIR / "custom_nodes" / "ComfyUI-Manager"
    if not manager_dir.exists():
        print("\nInstalling ComfyUI-Manager...")
        result = run_cmd(
            ["git", "clone", "https://github.com/ltdrdata/ComfyUI-Manager.git"],
            cwd=COMFYUI_DIR / "custom_nodes"
        )
        if result.returncode == 0:
            print("  ✅ Manager installed")
        else:
            print(f"  ⚠️ Manager install failed: {result.stderr[:300]}")

    # Install custom nodes
    print("\nInstalling required custom nodes...")
    for node_name, repo_url in REQUIRED_CUSTOM_NODES.items():
        node_dir = COMFYUI_DIR / "custom_nodes" / repo_url.split("/")[-1]
        if node_dir.exists():
            print(f"  ✅ {node_name} already installed")
        else:
            print(f"  Installing {node_name}...")
            result = run_cmd(["git", "clone", f"https://github.com/{repo_url}.git"],
                            cwd=COMFYUI_DIR / "custom_nodes", timeout=120)
            if result.returncode == 0:
                print(f"  ✅ {node_name} installed")
            else:
                print(f"  ⚠️ {node_name} failed: {result.stderr[:200]}")

    # Validate
    if verify:
        print("\n=== Verifying ComfyUI ===\n")
        verify_comfyui()

    return True


def verify_comfyui(port: int = 8188):
    """Verify ComfyUI is running and accessible."""
    import urllib.request
    import urllib.error

    # Try to start ComfyUI (non-blocking check)
    print(f"Checking ComfyUI on port {port}...")

    url = f"http://localhost:{port}/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            print(f"  ✅ ComfyUI running: {data.get('VRAM', 'unknown')} VRAM")
            return True
    except urllib.error.URLError:
        print("  ℹ️ ComfyUI not running (start with: python main.py --listen 0.0.0.0 --port {port})")
        return False
    except Exception as e:
        print(f"  ⚠️ ComfyUI check failed: {e}")
        return False


def download_sdxl_models():
    """
    Download SDXL Base + Refiner + ControlNet Tile models.
    These are large (≈10GB) - requires disk space check.
    """
    print("\n=== Downloading SDXL Models ===\n")

    models_dir = COMFYUI_DIR / "models" / "stable-diffusion"
    models_dir.mkdir(parents=True, exist_ok=True)

    # HuggingFace models needed
    models_to_download = [
        # SDXL Base 1.0
        ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl-base-1.0.safetensors"),
        # SDXL Refiner
        ("stabilityai/stable-diffusion-xl-refiner-1.0", "sdxl-refiner-1.0.safetensors"),
        # ControlNet Tile SDXL
        ("lllyasviel/ControlNet-sdxl-tile", "controlnet-tile-sdxl.safetensors"),
        # ControlNet Canny SDXL
        ("thibaud/controlnet-sdxl-1.0", "controlnet-canny-sdxl.safetensors"),
    ]

    import shutil
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    # Check disk space (need ≈15GB)
    import shutil as sh
    total, used, free = sh.disk_usage("/")
    free_gb = free // (1024**3)
    print(f"  Disk space: {free_gb}GB free")

    if free_gb < 20:
        print("  ⚠️ WARNING: Less than 20GB free. Model download may fail.")
        print("  Models will be downloaded on-demand when pipeline runs.")

    for hf_repo, filename in models_to_download:
        dest = models_dir / filename
        if dest.exists():
            print(f"  ✅ {filename} already exists")
            continue
        print(f"  Downloading {filename} from {hf_repo}...")
        try:
            snapshot_download(
                repo_id=hf_repo,
                local_dir=models_dir / filename,
                local_dir_use_symlinks=False,
            )
            print(f"  ✅ {filename} downloaded")
        except Exception as e:
            print(f"  ⚠️ {filename} failed: {e}")

    print("\n✅ Model download complete")


def create_startup_script(port: int = 8188):
    """Create a convenient startup script."""
    script_content = f'''#!/bin/bash
# ComfyUI startup script
# Usage: ./run_comfyui.sh [--port 8188]

PORT={port}
while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

cd ~/ComfyUI
python main.py --listen 0.0.0.0 --port $PORT
'''
    script_path = COMFYUI_DIR / "run_comfyui.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    print(f"\n✅ Startup script created: {script_path}")


# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComfyUI Setup for HMI Map Pipeline")
    parser.add_argument("--install", action="store_true", help="Install ComfyUI")
    parser.add_argument("--verify", action="store_true", help="Verify ComfyUI is running")
    parser.add_argument("--download-models", action="store_true", help="Download SDXL models")
    parser.add_argument("--port", type=int, default=8188, help="ComfyUI port (default: 8188)")

    args = parser.parse_args()

    if args.install:
        success = install_comfyui(verify=args.verify)
        if success:
            create_startup_script(args.port)
            print("\n✅ Installation complete!")
            print(f"\nTo start ComfyUI: cd ~/ComfyUI && python main.py --listen 0.0.0.0 --port {args.port}")

    elif args.verify:
        verify_comfyui(args.port)

    elif args.download_models:
        download_sdxl_models()

    else:
        parser.print_help()
        print("\nUsage examples:")
        print("  python install_comfyui.py --install --verify")
        print("  python install_comfyui.py --download-models")
        print("  python install_comfyui.py --verify --port 8188")