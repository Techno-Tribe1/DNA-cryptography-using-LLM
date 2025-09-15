from config import Config
from utils.tokenizer import DNACryptoTokenizer
from models.dna_crypto_model import DNACryptoModel
from training.trainer import DNATrainer
from training.dataset import generate_training_data
import os
import random

def main():
    print("🧬" + "="*50 + "🧬")
    print("   DNA ENCRYPTION LLM TRAINING WITH GRAPHS")
    print("🧬" + "="*50 + "🧬")
    print()
    
    # Initialize configuration (auto-detects hardware)
    config = Config('auto')  # or specify 'minimal', 'standard', 'large'
    print(f"📊 Configuration selected: {config.config_type.upper()}")
    print(f"💻 Device: {config.device}")
    print(f"🧠 Estimated model size: {config._estimate_params():.1f}M parameters")
    print()
    
    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    
    # Initialize tokenizer
    print("1️⃣ Initializing tokenizer...")
    tokenizer = DNACryptoTokenizer()
    print(f"   📝 Text vocabulary: {tokenizer.text_vocab_size}")
    print(f"   🧬 DNA vocabulary: {tokenizer.dna_vocab_size}")
    
    # Initialize model
    print("\n2️⃣ Initializing DNA Crypto model...")
    model = DNACryptoModel(
        text_vocab_size=tokenizer.text_vocab_size,
        dna_vocab_size=tokenizer.dna_vocab_size,
        **config.model_config
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   🎯 Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Generate training data with train/validation split
    print("\n3️⃣ Generating training data...")
    all_texts = generate_training_data(config.training_config['num_samples'])
    
    # Split into train/validation (80/20) - Like in research papers
    random.shuffle(all_texts)
    split_idx = int(0.8 * len(all_texts))
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    print(f"   📚 Training samples: {len(train_texts)}")
    print(f"   ✅ Validation samples: {len(val_texts)}")
    
    # Initialize enhanced trainer with graphing capabilities
    print("\n4️⃣ Initializing enhanced trainer...")
    trainer = DNATrainer(model, tokenizer, config)
    
    # Start training with comprehensive metrics collection
    print("\n5️⃣ Starting training with graph generation...")
    print(f"   📈 Epochs: {config.training_config['epochs']}")
    print(f"   📦 Batch size: {config.training_config['batch_size']}")
    print(f"   🎯 Learning rate: {config.training_config['learning_rate']}")
    print(f"   📊 Metrics: Loss, Accuracy, Training Time")
    print("   🎨 Graphs: Will be generated after training")
    print("="*70)
    
    try:
        # Start training with validation
        trainer.train(train_texts, val_texts)
        
        print("\n🎉 SUCCESS! Training completed!")
        print("📊 Check 'saved_models/training_curves.png' for your graphs!")
        print("💾 Training data saved in 'training_history.json'")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_checkpoint("saved_models/interrupted_checkpoint.pth")
        trainer.plot_comprehensive_training_curves()
        print("📊 Partial training graphs generated!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
