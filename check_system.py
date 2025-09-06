import torch
import psutil
import platform

def check_system():
    print("=== System Information ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Check PyTorch and GPU
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Recommend configuration
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    print("\n=== Recommended Configuration ===")
    if ram_gb < 8:
        print("⚠️  LOW RAM: Use minimal config")
        config = "minimal"
    elif ram_gb < 16:
        print("✅ GOOD: Use standard config")
        config = "standard"
    else:
        print("🚀 EXCELLENT: Use large config")
        config = "large"
    
    return config

if __name__ == "__main__":
    recommended_config = check_system()