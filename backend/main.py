import string
import os
import base64
import hashlib

class SimpleDNACrypto:
    """
    Optimized DNA Cryptography System with Compressed Binary Keys
    - Text maps to DNA sequences internally
    - DNA sequences convert to binary keys
    - Binary keys are compressed using base64 encoding
    - XOR encryption layer added for additional security
    """
    
    def __init__(self, seed=42, encryption_key=None):
        # Set deterministic seed for consistent mapping
        import random
        random.seed(seed)
        
        # Use EXACTLY 64 characters to match 64 DNA triplets (4³ = 64)
        self.supported_chars = list(string.ascii_lowercase + string.ascii_uppercase + string.digits + " ,")[:64]
        
        # DNA bases and mapping
        self.dna_bases = ['A', 'T', 'G', 'C']
        
        # Generate all 64 possible DNA triplets
        self.dna_triplets = []
        for base1 in self.dna_bases:
            for base2 in self.dna_bases:
                for base3 in self.dna_bases:
                    self.dna_triplets.append(base1 + base2 + base3)
        
        # Sort for consistent mapping
        self.dna_triplets.sort()
        
        # Create deterministic character → DNA triplet mapping
        self.char_to_dna = {}
        self.dna_to_char = {}
        
        for i, char in enumerate(self.supported_chars):
            triplet = self.dna_triplets[i]
            self.char_to_dna[char] = triplet
            self.dna_to_char[triplet] = char
        
        # DNA nucleotide → Binary mapping (2 bits per nucleotide)
        self.dna_to_binary = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
        self.binary_to_dna = {'00': 'A', '01': 'T', '10': 'G', '11': 'C'}
        
        # Encryption key for XOR layer (default: hash of seed)
        if encryption_key is None:
            encryption_key = hashlib.sha256(str(seed).encode()).digest()
        self.encryption_key = encryption_key
        
        pass
        pass
        pass
        pass
    
    def _xor_encrypt(self, binary_string):
        """Apply XOR encryption to binary string"""
        key_bits = ''.join(format(b, '08b') for b in self.encryption_key)
        encrypted = ''
        for i, bit in enumerate(binary_string):
            key_bit = key_bits[i % len(key_bits)]
            encrypted += str(int(bit) ^ int(key_bit))
        return encrypted
    
    def _xor_decrypt(self, binary_string):
        """Apply XOR decryption to binary string (XOR is symmetric)"""
        return self._xor_encrypt(binary_string)  # XOR is its own inverse
    
    def _binary_to_base64(self, binary_string):
        """Convert binary string to base64 for compression"""
        # Pad binary to multiple of 8
        padding = (8 - len(binary_string) % 8) % 8
        binary_string += '0' * padding
        
        # Convert binary to bytes
        bytes_data = bytes(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))
        
        # Encode to base64
        base64_str = base64.b64encode(bytes_data).decode('ascii')
        return base64_str, padding
    
    def _base64_to_binary(self, base64_str, padding):
        """Convert base64 back to binary string"""
        # Decode base64 to bytes
        bytes_data = base64.b64decode(base64_str.encode('ascii'))
        
        # Convert bytes to binary
        binary_string = ''.join(format(b, '08b') for b in bytes_data)
        
        # Remove padding
        if padding > 0:
            binary_string = binary_string[:-padding]
        
        return binary_string
    
    def get_supported_chars(self):
        """Return list of supported characters"""
        return self.supported_chars.copy()
    
    def encrypt(self, text):
        """
        Encrypt text to compressed binary key using DNA sequence mapping
        Flow: Text → DNA Sequence → Binary → XOR Encrypt → Base64 Compress
        """
        # Filter unsupported characters (replace with space)
        filtered_text = ''.join(c if c in self.char_to_dna else ' ' for c in text)
        
        # Step 1: Map text to DNA sequence
        dna_sequence = ''.join(self.char_to_dna[char] for char in filtered_text)
        
        # Step 2: Convert DNA sequence to binary key (as string)
        binary_key = ''.join(self.dna_to_binary[nucleotide] for nucleotide in dna_sequence)
        
        # Step 3: Apply XOR encryption for additional security
        encrypted_binary = self._xor_encrypt(binary_key)
        
        # Step 4: Compress binary to base64 (reduces length by ~33%)
        compressed_key, padding = self._binary_to_base64(encrypted_binary)
        
        # Calculate compression ratio
        original_length = len(binary_key)
        compressed_length = len(compressed_key)
        compression_ratio = (1 - compressed_length / original_length) * 100 if original_length > 0 else 0
        
        return {
            'original_text': text,
            'filtered_text': filtered_text,
            'dna_sequence': dna_sequence,  # Internal mapping (for reference)
            'original_binary_key': binary_key,  # Original binary key (before XOR and compression)
            'encrypted_binary_key': encrypted_binary,  # Binary after XOR encryption
            'binary_key': compressed_key,  # Compressed and encrypted key (Base64)
            'original_binary_length': original_length,  # Original binary length
            'compressed_length': compressed_length,  # Compressed key length
            'compression_ratio': round(compression_ratio, 2),  # Compression percentage
            'padding': padding,  # Padding info for decryption
            'stats': {
                'text_length': len(text),
                'dna_length': len(dna_sequence),
                'binary_length': original_length,
                'compressed_binary_length': compressed_length,
                'triplets_used': len(dna_sequence) // 3,
                'size_reduction': f"{compression_ratio:.1f}%"
            }
        }
    
    def decrypt(self, compressed_key, padding=None):
        """
        Decrypt compressed binary key back to text using DNA sequence mapping
        Flow: Base64 Decompress → XOR Decrypt → Binary → DNA Sequence → Text
        
        Args:
            compressed_key: Base64 encoded compressed key (or old binary format for backward compatibility)
            padding: Padding value (auto-detected if not provided for base64 keys)
        """
        # Check if it's base64 format (contains A-Z, a-z, 0-9, +, /, =)
        is_base64 = all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in compressed_key)
        
        if is_base64 and len(compressed_key) > 0:
            # New compressed format
            try:
                # Step 1: Decompress base64 to binary
                if padding is None:
                    # Try to decode and calculate padding
                    try:
                        bytes_data = base64.b64decode(compressed_key.encode('ascii'))
                        binary_key = ''.join(format(b, '08b') for b in bytes_data)
                        # Calculate padding by checking if binary length is multiple of 6
                        remainder = len(binary_key) % 6
                        padding = remainder if remainder > 0 else 0
                    except:
                        padding = 0
                
                encrypted_binary = self._base64_to_binary(compressed_key, padding)
                
                # Step 2: Decrypt XOR layer
                binary_key = self._xor_decrypt(encrypted_binary)
            except Exception as e:
                raise ValueError(f"Invalid compressed key format: {e}")
        else:
            # Old binary format (backward compatibility)
            if not all(c in '01' for c in compressed_key):
                raise ValueError("Binary key must be base64 encoded or contain only '0' and '1'")
            binary_key = compressed_key
        
        # Ensure binary key length is multiple of 6 (3 nucleotides × 2 bits each)
        remainder = len(binary_key) % 6
        if remainder != 0:
            # Pad with zeros to complete the last triplet
            binary_key += '0' * (6 - remainder)
        
        # Step 3: Convert binary key to DNA sequence
        dna_sequence = ''
        for i in range(0, len(binary_key), 2):
            binary_pair = binary_key[i:i+2]
            dna_sequence += self.binary_to_dna[binary_pair]
        
        # Step 4: Convert DNA sequence to text
        text = ''
        for i in range(0, len(dna_sequence), 3):
            triplet = dna_sequence[i:i+3]
            if len(triplet) == 3:  # Complete triplet
                text += self.dna_to_char.get(triplet, '?')
        
        return {
            'compressed_key': compressed_key if is_base64 else None,
            'binary_key': binary_key,
            'dna_sequence': dna_sequence,  # Internal mapping (for reference)
            'decrypted_text': text,
            'stats': {
                'compressed_length': len(compressed_key) if is_base64 else None,
                'binary_length': len(binary_key),
                'dna_length': len(dna_sequence),
                'text_length': len(text)
            }
        }
    
    def save_system(self, filepath="dna_crypto_system.pkl"):
        """Save the cryptographic system"""
        import pickle
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 System saved to: {filepath}")
    
    @classmethod
    def load_system(cls, filepath="dna_crypto_system.pkl"):
        """Load saved cryptographic system"""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def interactive_dna_binary_crypto():
    """Interactive DNA-based binary key cryptography system"""
    print("🧬 DNA-Based Binary Key Cryptography System 🔑")
    print("=" * 60)
    
    # Initialize the system
    crypto = SimpleDNACrypto()
    
    print(f"\n🔬 How it works:")
    print(f"   Encryption: Text → DNA Sequence → Binary → XOR Encrypt → Base64 Compress")
    print(f"   Decryption: Base64 Decompress → XOR Decrypt → Binary → DNA Sequence → Text")
    print(f"   The DNA sequence is the cryptographic mapping!")
    print(f"   ⚡ Binary keys are now compressed (~33% smaller) and encrypted!")
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU:")
        print("1.ENCRYPT text to binary key")
        print("2. DECRYPT binary key to text")
        print("3. Show supported characters")
        print("4. Run test suite")
        # print("5. Save system")
        print("5. Show DNA mapping (first 10)")
        print("6. EXIT")
        
        try:
            choice = input("\n👉 Enter choice (1-6): ").strip()
            
            if choice == '1':
                # ENCRYPT TEXT TO BINARY KEY
                print("\n🔒 ENCRYPTION MODE")
                text = input(" Enter text to encrypt: ")
                
                if not text:
                    print(" Empty text!")
                    continue
                
                result = crypto.encrypt(text)
                
                print(f"\n ENCRYPTION SUCCESSFUL!")
                print(f" Original text: '{result['original_text']}'")
                
                if result['original_text'] != result['filtered_text']:
                    print(f"  Filtered text: '{result['filtered_text']}' (unsupported chars → space)")
                
                print(f" DNA sequence: {result['dna_sequence']}")
                print(f" ORIGINAL BINARY KEY: {result['original_binary_key']}")
                print(f" COMPRESSED KEY (Base64): {result['binary_key']}")
                print(f" Original binary length: {result['original_binary_length']} bits")
                print(f" Compressed length: {result['compressed_length']} chars")
                print(f" Size reduction: {result['compression_ratio']}%")
                print(f" Stats: {result['stats']}")
                print(f"\n Use the compressed key (Base64) for decryption!")
                
            elif choice == '2':
                # DECRYPT BINARY KEY TO TEXT
                print("\n🔓 DECRYPTION MODE")
                print("   Accepts: Base64 compressed keys (new format) or binary keys (old format)")
                binary_key = input(" Enter key (Base64 or binary): ").strip()
                
                if not binary_key:
                    print("❌ Empty key!")
                    continue
                
                try:
                    result = crypto.decrypt(binary_key)
                    
                    print(f"\n✅ DECRYPTION SUCCESSFUL!")
                    if result['compressed_key']:
                        print(f" Compressed key: {result['compressed_key']}")
                    print(f" DNA sequence: {result['dna_sequence']}")
                    print(f" DECRYPTED TEXT: '{result['decrypted_text']}'")
                    print(f" Stats: {result['stats']}")
                    
                except ValueError as e:
                    print(f"❌ Error: {e}")
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                
            elif choice == '3':
                # SHOW SUPPORTED CHARACTERS
                chars = crypto.get_supported_chars()
                print(f"\n SUPPORTED CHARACTERS ({len(chars)}):")
                print(''.join(chars))
                print(f"\n Total: {len(chars)} characters mapped to {len(crypto.dna_triplets)} DNA triplets")
                
            elif choice == '4':
                # RUN TEST SUITE
                print("\n RUNNING TEST SUITE...")
                
                test_cases = [
                    "Hello World",
                    "DNA Crypto 123",
                    "Binary Key Test",
                    "ATGC nucleotides",
                    "Special chars ,"
                ]
                
                all_passed = True
                for i, test_text in enumerate(test_cases, 1):
                    try:
                        # Encrypt
                        encrypt_result = crypto.encrypt(test_text)
                        binary_key = encrypt_result['binary_key']
                        
                        # Decrypt
                        decrypt_result = crypto.decrypt(binary_key)
                        decrypted_text = decrypt_result['decrypted_text']
                        
                        # Check match (compare with filtered text)
                        expected = encrypt_result['filtered_text']
                        match = (expected == decrypted_text)
                        
                        status = "✅" if match else "❌"
                        print(f"Test {i}: '{test_text}' → {status}")
                        if match:
                            print(f"   Key length: {len(binary_key)} chars (compressed)")
                        
                        if not match:
                            all_passed = False
                            print(f"   Expected: '{expected}'")
                            print(f"   Got:      '{decrypted_text}'")
                        
                    except Exception as e:
                        print(f"Test {i}: '{test_text}' → ❌ Error: {e}")
                        all_passed = False
                
                print(f"\n RESULT: {' ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED'}")
                
            # elif choice == '5':
            #     # SAVE SYSTEM
            #     filepath = input("\n Enter save path (default: dna_crypto.pkl): ").strip()
            #     if not filepath:
            #         filepath = "dna_crypto.pkl"
            #     try:
            #         crypto.save_system(filepath)
            #     except Exception as e:
            #         print(f" Save error: {e}")
                
            # elif choice == '6':
            #     # LOAD SYSTEM
            #     filepath = input("\n Enter load path (default: dna_crypto.pkl): ").strip()
            #     if not filepath:
            #         filepath = "dna_crypto.pkl"
            #     try:
            #         crypto = DNABinaryKeyCrypto.load_system(filepath)
            #         print(f" System loaded from: {filepath}")
            #     except FileNotFoundError:
            #         print(f" File not found: {filepath}")
            #     except Exception as e:
            #         print(f" Load error: {e}")
                
            elif choice == '5':
                # SHOW DNA MAPPING
                print(f"\n🔬 DNA MAPPING (First 10 characters):")
                for i, (char, dna) in enumerate(list(crypto.char_to_dna.items())[:10]):
                    binary = ''.join(crypto.dna_to_binary[n] for n in dna)
                    print(f"'{char}' → {dna} → {binary}")
                
                print(f"\n🧬 DNA to Binary mapping:")
                for dna, binary in crypto.dna_to_binary.items():
                    print(f"{dna} → {binary}")
                
            elif choice == '6':
                # EXIT
                print("\n Thank you for using DNA Binary Crypto!")
                break
                
            else:
                print(" Invalid choice! Please enter 1-8.")
        
        except KeyboardInterrupt:
            print("\n\n Exiting...")
            break
        except Exception as e:
            print(f" Unexpected error: {e}")


def quick_demo():
    """Quick demonstration"""
    print(" Quick DNA Binary Key Demo")
    print("=" * 40)
    
    crypto = SimpleDNACrypto()
    
    test_messages = [
        "Hello World",
        "DNA Crypto",
        "Binary Key 123"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n🧪 Test {i}: '{message}'")
        
        # Encrypt to binary key
        encrypt_result = crypto.encrypt(message)
        binary_key = encrypt_result['binary_key']
        
        # Decrypt from binary key
        decrypt_result = crypto.decrypt(binary_key)
        decrypted = decrypt_result['decrypted_text']
        
        # Show results
        print(f" Binary Key: {binary_key[:50]}{'...' if len(binary_key) > 50 else ''}")
        print(f" Decrypted:  '{decrypted}'")
        print(f" Match: {encrypt_result['filtered_text'] == decrypted}")


if __name__ == "__main__":
    print(" DNA-Based Binary Key Cryptography ")
    print()
    
    choice = input("Select mode:\n1.  Interactive system\n2.  Quick demo\n\nChoice (1-2): ").strip()
    
    if choice == '1':
        interactive_dna_binary_crypto()
    elif choice == '2':
        quick_demo()
        if input("\nStart interactive system? (y/n): ").lower().startswith('y'):
            interactive_dna_binary_crypto()
    else:
        print("Invalid choice. Starting interactive system...")
        interactive_dna_binary_crypto()