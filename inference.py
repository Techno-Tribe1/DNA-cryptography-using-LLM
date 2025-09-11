import torch
import os
from models.dna_crypto_model import DNACryptoModel

class DNAEncryptor:
    """Easy-to-use interface for trained DNA encryption model"""
    
    def __init__(self, model_path: str = "saved_models/final_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        
        self.tokenizer = checkpoint['tokenizer']
        
        # Recreate model with saved config
        self.model = DNACryptoModel(
            text_vocab_size=self.tokenizer.text_vocab_size,
            dna_vocab_size=self.tokenizer.dna_vocab_size,
            **checkpoint['model_config']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ DNA Encryptor loaded from {model_path}")
        print(f"Device: {self.device}")
    
    def encrypt(self, text: str) -> dict:
        """Encrypt text and return comprehensive results"""
        with torch.no_grad():
            # Generate DNA sequence
            dna_sequence = self._text_to_dna(text)
            
            # Convert to binary
            binary_data = self._dna_to_binary(dna_sequence)
            
            return {
                'original': text,
                'dna_sequence': dna_sequence,
                'binary_data': binary_data,
                'stats': {
                    'text_length': len(text),
                    'dna_length': len(dna_sequence),
                    'binary_length': len(binary_data),
                    'compression_ratio': len(binary_data) / (len(text) * 8) if text else 0
                }
            }
    
    def decrypt(self, dna_sequence: str = None, binary_data: str = None) -> str:
        """Decrypt from DNA sequence or binary data"""
        with torch.no_grad():
            if binary_data and not dna_sequence:
                dna_sequence = self._binary_to_dna(binary_data)
            
            if not dna_sequence:
                raise ValueError("Either dna_sequence or binary_data must be provided")
            
            return self._dna_to_text(dna_sequence)
    
    def _text_to_dna(self, text: str) -> str:
        """Convert text to DNA using trained model"""
        # Tokenize input
        text_tokens = self.tokenizer.encode_text(text, self.model.max_text_len).unsqueeze(0).to(self.device)
        
        # Generate DNA sequence
        dna_sequence = torch.tensor([[self.tokenizer.dna_start_id]]).to(self.device)
        
        for _ in range(self.model.max_dna_len - 1):
            output = self.model.encrypt_forward(
                text_tokens, dna_sequence,
                self.tokenizer.pad_id, self.tokenizer.dna_pad_id
            )
            
            # Get next token (greedy decoding)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            dna_sequence = torch.cat([dna_sequence, next_token], dim=1)
            
            # Stop if end token generated
            if next_token.item() == self.tokenizer.dna_end_id:
                break
        
        return self.tokenizer.decode_dna(dna_sequence[0])
    
    def _dna_to_text(self, dna_sequence: str) -> str:
        """Convert DNA to text using trained model"""
        # Tokenize DNA
        dna_tokens = self.tokenizer.encode_dna(dna_sequence, self.model.max_dna_len).unsqueeze(0).to(self.device)
        
        # Generate text sequence
        text_sequence = torch.tensor([[self.tokenizer.start_id]]).to(self.device)
        
        for _ in range(self.model.max_text_len - 1):
            output = self.model.decrypt_forward(
                dna_tokens, text_sequence,
                self.tokenizer.dna_pad_id, self.tokenizer.pad_id
            )
            
            # Get next token
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            text_sequence = torch.cat([text_sequence, next_token], dim=1)
            
            # Stop if end token generated
            if next_token.item() == self.tokenizer.end_id:
                break
        
        return self.tokenizer.decode_text(text_sequence[0])
    
    def _dna_to_binary(self, dna_sequence: str) -> str:
        """Convert DNA to binary"""
        mapping = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
        return ''.join(mapping.get(nuc, '00') for nuc in dna_sequence)
    
    def _binary_to_dna(self, binary_data: str) -> str:
        """Convert binary to DNA"""
        mapping = {'00': 'A', '01': 'T', '10': 'G', '11': 'C'}
        dna = ''
        for i in range(0, len(binary_data), 2):
            if i + 1 < len(binary_data):
                pair = binary_data[i:i+2]
                dna += mapping.get(pair, 'A')
        return dna

def demo_trained_model():
    """Demo the trained model"""
    print("=== DNA Encryption LLM Demo ===\n")
    
    try:
        # Load trained model
        encryptor = DNAEncryptor()
        
        # Test cases
        test_messages = [
            "Hello World!",
            "Secret password: admin123",
            "Transfer $5000 to account ABC123",
            "API_KEY=xyz789def456",
            "Meeting scheduled for 3 PM tomorrow"
        ]
        
        print("Testing trained DNA encryption model:")
        print("=" * 60)
        
        for i, message in enumerate(test_messages, 1):
            print(f"\nTest {i}: '{message}'")
            
            try:
                # Encrypt
                result = encryptor.encrypt(message)
                dna_seq = result['dna_sequence']
                binary = result['binary_data']
                
                print(f"DNA:    {dna_seq[:40]}{'...' if len(dna_seq) > 40 else ''}")
                print(f"Binary: {binary[:40]}{'...' if len(binary) > 40 else ''}")
                
                # Decrypt
                decrypted = encryptor.decrypt(dna_sequence=dna_seq)
                print(f"Result: '{decrypted}'")
                
                # Check accuracy
                match = message.lower().strip() in decrypted.lower().strip()
                print(f"Status: {'✅ Good' if match else '❌ Poor'}")
                print(f"Stats:  {result['stats']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except FileNotFoundError:
        print("❌ No trained model found!")
        print("Please run 'python train.py' first to train a model.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def interactive_trained():
    """Interactive mode with trained model"""
    try:
        encryptor = DNAEncryptor()
        
        print("\n=== Interactive DNA Encryption (Trained Model) ===")
        print("Commands: encrypt, decrypt, quit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'encrypt':
                    text = input("Enter text to encrypt: ")
                    result = encryptor.encrypt(text)
                    print(f"DNA: {result['dna_sequence']}")
                    print(f"Binary: {result['binary_data']}")
                    print(f"Stats: {result['stats']}")
                    
                elif command == 'decrypt':
                    choice = input("Decrypt from (d)na or (b)inary? ").lower()
                    if choice == 'd':
                        dna = input("Enter DNA sequence: ")
                        decrypted = encryptor.decrypt(dna_sequence=dna)
                    elif choice == 'b':
                        binary = input("Enter binary data: ")
                        decrypted = encryptor.decrypt(binary_data=binary)
                    else:
                        print("Invalid choice!")
                        continue
                    print(f"Decrypted: '{decrypted}'")
                    
                else:
                    print("Unknown command! Use: encrypt, decrypt, quit")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    except FileNotFoundError:
        print("❌ No trained model found! Please train a model first.")

if __name__ == "__main__":
    demo_trained_model()
    
    if input("\nStart interactive mode? (y/n): ").lower() == 'y':
        interactive_trained()
