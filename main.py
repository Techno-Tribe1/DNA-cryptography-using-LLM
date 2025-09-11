from models.simple_crypto import SimpleDNACrypto
import time

def demo_simple_crypto():
    """Quick demo of simple DNA encryption"""
    print("=== Simple DNA Encryption Demo ===\n")
    
    # Initialize
    crypto = SimpleDNACrypto()
    
    # Test cases
    test_messages = [
        "Hello World!",
        "Password: admin123",
        "Transfer $1000 to account XYZ789",
        "API_KEY=abcdef123456",
        "This is a longer message to test encryption efficiency and accuracy."
    ]
    
    print("Testing Simple DNA Encryption:")
    print("=" * 80)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}:")
        print(f"Original:  '{message}'")
        
        # Encrypt
        start_time = time.time()
        result = crypto.encrypt(message)
        encrypt_time = (time.time() - start_time) * 1000
        
        # Show results
        dna_seq = result['dna_sequence']
        binary_data = result['binary_data']
        
        print(f"DNA:       {dna_seq[:50]}{'...' if len(dna_seq) > 50 else ''}")
        print(f"Binary:    {binary_data[:50]}{'...' if len(binary_data) > 50 else ''}")
        
        # Decrypt
        start_time = time.time()
        decrypted = crypto.decrypt(dna_sequence=dna_seq)
        decrypt_time = (time.time() - start_time) * 1000
        
        print(f"Decrypted: '{decrypted}'")
        
        # Verification
        match = message == decrypted
        print(f"✅ Match: {match}")
        print(f"⚡ Speed: Encrypt {encrypt_time:.2f}ms, Decrypt {decrypt_time:.2f}ms")
        print(f"📊 Stats: {result['stats']}")
    
    # Save the system
    crypto.save('saved_models/simple_crypto.pkl')
    print(f"\n💾 Simple crypto system saved to saved_models/simple_crypto.pkl")

def interactive_simple():
    """Interactive mode for simple encryption"""
    crypto = SimpleDNACrypto()
    
    print("\n=== Interactive DNA Encryption ===")
    print("Commands: encrypt, decrypt, save, load, quit")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'encrypt':
                text = input("Enter text to encrypt: ")
                result = crypto.encrypt(text)
                print(f"DNA: {result['dna_sequence']}")
                print(f"Binary: {result['binary_data']}")
                print(f"Stats: {result['stats']}")
                
            elif command == 'decrypt':
                choice = input("Decrypt from (d)na or (b)inary? ").lower()
                if choice == 'd':
                    dna = input("Enter DNA sequence: ")
                    decrypted = crypto.decrypt(dna_sequence=dna)
                elif choice == 'b':
                    binary = input("Enter binary data: ")
                    decrypted = crypto.decrypt(binary_data=binary)
                else:
                    print("Invalid choice!")
                    continue
                print(f"Decrypted: '{decrypted}'")
                
            elif command == 'save':
                filepath = input("Enter save path (default: simple_crypto.pkl): ") or "simple_crypto.pkl"
                crypto.save(filepath)
                print(f"Saved to {filepath}")
                
            elif command == 'load':
                filepath = input("Enter load path (default: simple_crypto.pkl): ") or "simple_crypto.pkl"
                crypto = SimpleDNACrypto.load(filepath)
                print(f"Loaded from {filepath}")
                
            else:
                print("Unknown command! Use: encrypt, decrypt, save, load, quit")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    demo_simple_crypto()
    
    # Ask if user wants interactive mode
    if input("\nStart interactive mode? (y/n): ").lower() == 'y':
        interactive_simple()