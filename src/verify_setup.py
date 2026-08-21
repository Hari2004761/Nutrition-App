import sys

print(f"Python: {sys.version}")

# --- PyTorch / CUDA ---
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        c = torch.matmul(a, b)
        print(f"Matmul result shape: {c.shape}  — GPU OK")
    else:
        print("CUDA not detected — check torch build and driver version.")
        sys.exit(1)
except ImportError as e:
    print(f"PyTorch import failed: {e}")
    sys.exit(1)

# --- Other packages ---
for pkg in ["ultralytics", "streamlit", "matplotlib", "pandas", "requests", "PIL"]:
    try:
        __import__(pkg)
        print(f"{pkg}: OK")
    except ImportError as e:
        print(f"{pkg}: MISSING — {e}")

print("\nAll checks passed.")
