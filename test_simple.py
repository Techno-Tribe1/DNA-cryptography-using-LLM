import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_hardware_check():
    """Check if system is ready"""
    print("=== Hardware Check ===")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed!")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not installed!")
        return False
    
    # Check memory
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"✅ RAM: {ram_gb:.1f} GB")
        
        if ram_gb < 4:
            print("⚠️  Warning: Low RAM. Use minimal configuration.")
        elif ram_gb < 8:
            print("✅ Good: Standard configuration recommended.")
        else:
            print("🚀 Excellent: Large configuration possible.")
            
    except ImportError:
        print("⚠️  psutil not available, cannot check RAM")
    
    return True

def test_simple_crypto():
    """Test simple DNA crypto"""
    print("\n=== Testing Simple DNA Crypto ===")
    
    try:
        from models.simple_crypto import SimpleDNACrypto
        
        crypto = SimpleDNACrypto()
        
        # Test basic functionality
        test_message = "Hello World!"
        print(f"Testing: '{test_message}'")
        
        # Encrypt
        result = crypto.encrypt(test_message)
        print(f"✅ Encryption successful")
        print(f"   DNA length: {len(result['dna_sequence'])}")
        print(f"   Binary length: {len(result['binary_data'])}")
        
        # Decrypt
        decrypted = crypto.decrypt(dna_sequence=result['dna_sequence'])
        print(f"✅ Decryption successful: '{decrypted}'")
        
        # Verify
        if test_message == decrypted:
            print("🎉 Perfect match! Simple crypto working correctly.")
            return True
        else:
            print("❌ Mismatch! Something went wrong.")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer():
    """Test tokenizer"""
    print("\n=== Testing Tokenizer ===")
    
    try:
        from utils.tokenizer import DNACryptoTokenizer
        
        tokenizer = DNACryptoTokenizer()
        
        # Test text encoding/decoding
        test_text = "Hello!"
        tokens = tokenizer.encode_text(test_text, 32)
        decoded = tokenizer.decode_text(tokens)
        
        print(f"Text: '{test_text}' -> '{decoded}'")
        
        if test_text.lower() in decoded.lower():
            print("✅ Text tokenizer working")
        else:
            print("⚠️  Text tokenizer may have issues")
        
        # Test DNA encoding/decoding
        test_dna = "ATGCATGC"
        dna_tokens = tokenizer.encode_dna(test_dna, 32)
        decoded_dna = tokenizer.decode_dna(dna_tokens)
        
        print(f"DNA: '{test_dna}' -> '{decoded_dna}'")
        
        if test_dna in decoded_dna:
            print("✅ DNA tokenizer working")
            return True
        else:
            print("⚠️  DNA tokenizer may have issues")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("DNA Encryption LLM - System Test")
    print("=" * 40)
    
    # Run tests
    hardware_ok = quick_hardware_check()
    
    if hardware_ok:
        simple_ok = test_simple_crypto()
        tokenizer_ok = test_tokenizer()
        
        print("\n" + "=" * 40)
        print("SUMMARY:")
        print(f"Hardware: {'✅' if hardware_ok else '❌'}")
        print(f"Simple Crypto: {'✅' if simple_ok else '❌'}")
        print(f"Tokenizer: {'✅' if tokenizer_ok else '❌'}")
        
        if all([hardware_ok, simple_ok, tokenizer_ok]):
            print("\n🎉 All systems ready! You can proceed with:")
            print("   python main.py        (for simple crypto)")
            print("   python train.py       (for LLM training)")
        else:
            print("\n⚠️  Some issues detected. Check the errors above.")
    else:
        print("\n❌ Hardware requirements not met. Please install required packages.")
