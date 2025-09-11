from config import Config
from utils.tokenizer import DNACryptoTokenizer
from models.dna_crypto_model import DNACryptoModel
from training.trainer import DNATrainer
from training.dataset import generate_training_data
import os

def main():
    print("=== DNA Encryption LLM Training ===\n")
    
    # Initialize configuration (auto-detects hardware)
    config = Config('auto')  # or specify 'minimal', 'standard', 'large'
    
    print(f"Configuration selected: {config.config_type}")
    print(f"Device: {config.device}")
    print(f"Estimated model size: {config._estimate_params():.1f}M parameters\n")
    
    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    
    # Initialize tokenizer
    print("1. Initializing tokenizer...")
    tokenizer = DNACryptoTokenizer()
    print(f"   Text vocabulary: {tokenizer.text_vocab_size}")
    print(f"   DNA vocabulary: {tokenizer.dna_vocab_size}")
    
    # Initialize model
    print("\n2. Initializing model...")
    model = DNACryptoModel(
        text_vocab_size=tokenizer.text_vocab_size,
        dna_vocab_size=tokenizer.dna_vocab_size,
        **config.model_config
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Generate training data
    print("\n3. Generating training data...")
    train_texts = generate_training_data(config.training_config['num_samples'])
    print(f"   Generated {len(train_texts)} samples")
    
    # Initialize trainer
    print("\n4. Initializing trainer...")
    trainer = DNATrainer(model, tokenizer, config)
    
    # Start training
    print("\n5. Starting training...")
    print(f"   Epochs: {config.training_config['epochs']}")
    print(f"   Batch size: {config.training_config['batch_size']}")
    print(f"   Learning rate: {config.training_config['learning_rate']}")
    print("-" * 50)
    
    try:
        trainer.train(train_texts)
        print("\n✅ Training completed successfully!")
        
        # Test the trained model
        print("\n6. Testing trained model...")
        test_model(trainer, tokenizer)
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_checkpoint("saved_models/interrupted_checkpoint.pth")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def test_model(trainer, tokenizer):
    """Quick test of trained model"""
    from models.dna_crypto_system import DNACryptoSystem
    
    # Create crypto system
    crypto_system = DNACryptoSystem(trainer.model, tokenizer)
    
    test_texts = [
        "Hello World",
        "Secret message",
        "Password123",
        "API key: abc123"
    ]
    
    print("Testing encryption/decryption:")
    for text in test_texts:
        try:
            # Encrypt
            dna_seq, binary = crypto_system.encrypt_text(text)
            # Decrypt
            decrypted = crypto_system.decrypt_text(dna_sequence=dna_seq)
            
            match = text.lower().strip() == decrypted.lower().strip()
            print(f"'{text}' -> '{decrypted}' | Match: {match}")
            
        except Exception as e:
            print(f"'{text}' -> Error: {e}")

class DNACryptoSystem:
    """Inference system for trained model"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.model.eval()
    
    def encrypt_text(self, text: str):
        """Encrypt text to DNA and binary"""
        with torch.no_grad():
            # Tokenize text
            text_tokens = self.tokenizer.encode_text(text, self.model.max_text_len).unsqueeze(0).to(self.device)
            
            # Generate DNA sequence
            dna_seq = self._generate_dna(text_tokens)
            
            # Convert to binary
            binary = self._dna_to_binary(dna_seq)
            
            return dna_seq, binary
    
    def decrypt_text(self, dna_sequence=None, binary_data=None):
        """Decrypt DNA/binary back to text"""
        with torch.no_grad():
            if binary_data and not dna_sequence:
                dna_sequence = self._binary_to_dna(binary_data)
            
            # Tokenize DNA
            dna_tokens = self.tokenizer.encode_dna(dna_sequence, self.model.max_dna_len).unsqueeze(0).to(self.device)
            
            # Generate text
            text = self._generate_text(dna_tokens)
            
            return text
    
    def _generate_dna(self, text_tokens):
        """Generate DNA from text tokens"""
        batch_size = text_tokens.size(0)
        max_len = self.model.max_dna_len
        
        # Start with start token
        dna_seq = torch.tensor([[self.tokenizer.dna_start_id]] * batch_size).to(self.device)
        
        for _ in range(max_len - 1):
            output = self.model.encrypt_forward(
                text_tokens, dna_seq,
                self.tokenizer.pad_id, self.tokenizer.dna_pad_id
            )
            
            # Get next token
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            dna_seq = torch.cat([dna_seq, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.dna_end_id:
                break
        
        return self.tokenizer.decode_dna(dna_seq[0])
    
    def _generate_text(self, dna_tokens):
        """Generate text from DNA tokens"""
        batch_size = dna_tokens.size(0)
        max_len = self.model.max_text_len
        
        # Start with start token
        text_seq = torch.tensor([[self.tokenizer.start_id]] * batch_size).to(self.device)
        
        for _ in range(max_len - 1):
            output = self.model.decrypt_forward(
                dna_tokens, text_seq,
                self.tokenizer.dna_pad_id, self.tokenizer.pad_id
            )
            
            # Get next token
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            text_seq = torch.cat([text_seq, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.end_id:
                break
        
        return self.tokenizer.decode_text(text_seq[0])
    
    def _dna_to_binary(self, dna_sequence):
        """Convert DNA to binary"""
        mapping = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
        return ''.join(mapping.get(nuc, '00') for nuc in dna_sequence)
    
    def _binary_to_dna(self, binary_data):
        """Convert binary to DNA"""
        mapping = {'00': 'A', '01': 'T', '10': 'G', '11': 'C'}
        dna = ''
        for i in range(0, len(binary_data), 2):
            if i + 1 < len(binary_data):
                pair = binary_data[i:i+2]
                dna += mapping.get(pair, 'A')
        return dna

if __name__ == "__main__":
    main()
