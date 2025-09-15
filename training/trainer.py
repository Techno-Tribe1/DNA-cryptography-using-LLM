import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
import numpy as np

class DNATrainer:
    """Enhanced Training pipeline with comprehensive metrics tracking"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.model.to(self.device)
        
        # Training setup
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training_config['learning_rate']
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
        
        # Metrics tracking - Similar to the paper's results
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_encrypt_loss': [],
            'train_decrypt_loss': [],
            'val_encrypt_loss': [],
            'val_decrypt_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        print(f"🧬 Enhanced DNA Crypto Trainer initialized on {self.device}")
        print(f"📊 Metrics tracking enabled - Graphs will be generated!")

    def train(self, train_texts, val_texts=None):
        """Main training loop with comprehensive metrics collection"""
        from training.dataset import DNACryptoDataset
        
        # Create training dataset
        train_dataset = DNACryptoDataset(
            train_texts,
            self.tokenizer,
            self.config.model_config['max_text_len'],
            self.config.model_config['max_dna_len']
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training_config['batch_size'],
            shuffle=True
        )
        
        # Create validation dataset if provided
        val_dataloader = None
        if val_texts:
            val_dataset = DNACryptoDataset(
                val_texts,
                self.tokenizer,
                self.config.model_config['max_text_len'],
                self.config.model_config['max_dna_len']
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.training_config['batch_size'],
                shuffle=False
            )
        
        print("🚀 Starting DNA Encryption LLM Training with Metrics...")
        print("=" * 70)
        
        for epoch in range(self.config.training_config['epochs']):
            epoch_start_time = datetime.now()
            
            # Training phase
            train_metrics = self._train_epoch(epoch, train_dataloader)
            
            # Validation phase
            val_metrics = {'val_loss': 0, 'val_encrypt_loss': 0, 'val_decrypt_loss': 0, 'val_accuracy': 0}
            if val_dataloader:
                val_metrics = self._validate_epoch(epoch, val_dataloader)
            
            # Record all metrics
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['train_encrypt_loss'].append(train_metrics['train_encrypt_loss'])
            self.training_history['train_decrypt_loss'].append(train_metrics['train_decrypt_loss'])
            self.training_history['train_accuracy'].append(train_metrics['train_accuracy'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['val_encrypt_loss'].append(val_metrics['val_encrypt_loss'])
            self.training_history['val_decrypt_loss'].append(val_metrics['val_decrypt_loss'])
            self.training_history['val_accuracy'].append(val_metrics['val_accuracy'])
            
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            self.training_history['epoch_times'].append(epoch_time)
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            self._print_epoch_summary(epoch + 1, train_metrics, val_metrics, epoch_time)
            
            # Save checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(f"saved_models/checkpoint_epoch_{epoch+1}.pth")
        
        # Save final model and generate comprehensive analysis
        self.save_model("saved_models/final_model.pth")
        self.plot_comprehensive_training_curves()
        self.save_training_history()
        self.print_training_summary()
        
        print("\n✅ Training completed successfully!")
        print("📊 Professional training graphs saved to 'saved_models/training_curves.png'")
        print("💾 Training metrics saved to 'training_history.json'")

    def _train_epoch(self, epoch, dataloader):
        """Train single epoch with detailed metrics collection"""
        self.model.train()
        total_loss = 0
        encrypt_losses = []
        decrypt_losses = []
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            text_tokens = batch['text_tokens'].to(self.device)
            dna_tokens = batch['dna_tokens'].to(self.device)
            
            # Prepare input/target pairs
            text_input = text_tokens[:, :-1]
            text_target = text_tokens[:, 1:]
            dna_input = dna_tokens[:, :-1]
            dna_target = dna_tokens[:, 1:]
            
            self.optimizer.zero_grad()
            
            # Encryption forward pass
            encrypt_output = self.model.encrypt_forward(
                text_tokens, dna_input,
                self.tokenizer.pad_id, self.tokenizer.dna_pad_id
            )
            loss_encrypt = self.criterion(
                encrypt_output.reshape(-1, encrypt_output.size(-1)),
                dna_target.reshape(-1)
            )
            
            # Decryption forward pass
            decrypt_output = self.model.decrypt_forward(
                dna_tokens, text_input,
                self.tokenizer.dna_pad_id, self.tokenizer.pad_id
            )
            loss_decrypt = self.criterion(
                decrypt_output.reshape(-1, decrypt_output.size(-1)),
                text_target.reshape(-1)
            )
            
            # Combined loss
            total_batch_loss = loss_encrypt + loss_decrypt
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track losses and accuracy
            total_loss += total_batch_loss.item()
            encrypt_losses.append(loss_encrypt.item())
            decrypt_losses.append(loss_decrypt.item())
            
            # Calculate accuracy (approximate)
            predictions = torch.argmax(encrypt_output, dim=-1)
            correct_predictions += (predictions == dna_target).sum().item()
            total_predictions += dna_target.numel()
            
            # Progress reporting
            if batch_idx % 10 == 0:
                progress = (batch_idx / len(dataloader)) * 100
                print(f"🔄 Epoch {epoch+1}/{self.config.training_config['epochs']} | "
                      f"Progress: {progress:.1f}% | "
                      f"Loss: {total_batch_loss.item():.4f} | "
                      f"Encrypt: {loss_encrypt.item():.4f} | "
                      f"Decrypt: {loss_decrypt.item():.4f}")
        
        # Calculate epoch averages
        avg_loss = total_loss / len(dataloader)
        avg_encrypt = sum(encrypt_losses) / len(encrypt_losses)
        avg_decrypt = sum(decrypt_losses) / len(decrypt_losses)
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        return {
            'train_loss': avg_loss,
            'train_encrypt_loss': avg_encrypt,
            'train_decrypt_loss': avg_decrypt,
            'train_accuracy': accuracy
        }

    def _validate_epoch(self, epoch, dataloader):
        """Validate single epoch"""
        self.model.eval()
        total_loss = 0
        encrypt_losses = []
        decrypt_losses = []
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                text_tokens = batch['text_tokens'].to(self.device)
                dna_tokens = batch['dna_tokens'].to(self.device)
                
                text_input = text_tokens[:, :-1]
                text_target = text_tokens[:, 1:]
                dna_input = dna_tokens[:, :-1]
                dna_target = dna_tokens[:, 1:]
                
                # Forward passes
                encrypt_output = self.model.encrypt_forward(
                    text_tokens, dna_input,
                    self.tokenizer.pad_id, self.tokenizer.dna_pad_id
                )
                loss_encrypt = self.criterion(
                    encrypt_output.reshape(-1, encrypt_output.size(-1)),
                    dna_target.reshape(-1)
                )
                
                decrypt_output = self.model.decrypt_forward(
                    dna_tokens, text_input,
                    self.tokenizer.dna_pad_id, self.tokenizer.pad_id
                )
                loss_decrypt = self.criterion(
                    decrypt_output.reshape(-1, decrypt_output.size(-1)),
                    text_target.reshape(-1)
                )
                
                total_batch_loss = loss_encrypt + loss_decrypt
                total_loss += total_batch_loss.item()
                encrypt_losses.append(loss_encrypt.item())
                decrypt_losses.append(loss_decrypt.item())
                
                # Calculate accuracy
                predictions = torch.argmax(encrypt_output, dim=-1)
                correct_predictions += (predictions == dna_target).sum().item()
                total_predictions += dna_target.numel()
        
        # Calculate averages
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_encrypt = sum(encrypt_losses) / len(encrypt_losses) if encrypt_losses else 0
        avg_decrypt = sum(decrypt_losses) / len(decrypt_losses) if decrypt_losses else 0
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        return {
            'val_loss': avg_loss,
            'val_encrypt_loss': avg_encrypt,
            'val_decrypt_loss': avg_decrypt,
            'val_accuracy': accuracy
        }

    def _print_epoch_summary(self, epoch, train_metrics, val_metrics, epoch_time):
        """Print detailed epoch summary"""
        print(f"\n📊 EPOCH {epoch} SUMMARY:")
        print(f"   🎯 Training Loss: {train_metrics['train_loss']:.4f}")
        print(f"   🔒 Encryption Loss: {train_metrics['train_encrypt_loss']:.4f}")
        print(f"   🔓 Decryption Loss: {train_metrics['train_decrypt_loss']:.4f}")
        print(f"   📈 Training Accuracy: {train_metrics['train_accuracy']:.2f}%")
        
        if val_metrics['val_loss'] > 0:
            print(f"   ✅ Validation Loss: {val_metrics['val_loss']:.4f}")
            print(f"   📊 Val Accuracy: {val_metrics['val_accuracy']:.2f}%")
        
        print(f"   ⏱️ Epoch Time: {epoch_time:.2f}s")
        print("   " + "="*50)

    def plot_comprehensive_training_curves(self):
        """Generate comprehensive training visualization - Similar to research paper"""
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Set style for professional publication-ready plots
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🧬 DNA Encryption LLM Training Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall Loss Comparison (Main Graph - Like in Paper)
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.training_history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
        if max(self.training_history['val_loss']) > 0:
            ax1.plot(epochs, self.training_history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        ax1.set_title('Training & Validation Loss', fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy Comparison (Like in Paper)
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.training_history['train_accuracy'], 'g-o', label='Training Accuracy', linewidth=2, markersize=4)
        if max(self.training_history['val_accuracy']) > 0:
            ax2.plot(epochs, self.training_history['val_accuracy'], 'orange', marker='s', label='Validation Accuracy', linewidth=2, markersize=4)
        ax2.set_title('Training & Validation Accuracy', fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Encryption vs Decryption Loss
        ax3 = axes[0, 2]
        ax3.plot(epochs, self.training_history['train_encrypt_loss'], 'purple', marker='o', label='Encryption Loss', linewidth=2)
        ax3.plot(epochs, self.training_history['train_decrypt_loss'], 'brown', marker='s', label='Decryption Loss', linewidth=2)
        ax3.set_title('Encryption vs Decryption Loss', fontweight='bold')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training Time Analysis
        ax4 = axes[1, 0]
        ax4.plot(epochs, self.training_history['epoch_times'], 'teal', marker='o', label='Epoch Time', linewidth=2)
        ax4.set_title('Training Time per Epoch', fontweight='bold')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Time (seconds)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Loss Reduction Progress
        ax5 = axes[1, 1]
        if self.training_history['train_loss']:
            initial_loss = self.training_history['train_loss'][0]
            loss_reduction = [(initial_loss - loss) / initial_loss * 100 for loss in self.training_history['train_loss']]
            ax5.plot(epochs, loss_reduction, 'red', marker='o', label='Loss Reduction %', linewidth=2)
        ax5.set_title('Loss Reduction Progress', fontweight='bold')
        ax5.set_xlabel('Epochs')
        ax5.set_ylabel('Loss Reduction (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Performance Summary
        ax6 = axes[1, 2]
        categories = ['Train Acc', 'Val Acc', 'Train Loss', 'Val Loss']
        final_train_acc = self.training_history['train_accuracy'][-1] if self.training_history['train_accuracy'] else 0
        final_val_acc = self.training_history['val_accuracy'][-1] if max(self.training_history['val_accuracy']) > 0 else 0
        final_train_loss = self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 0
        final_val_loss = self.training_history['val_loss'][-1] if max(self.training_history['val_loss']) > 0 else 0
        
        values = [final_train_acc, final_val_acc, final_train_loss * 100, final_val_loss * 100]  # Scale loss for visibility
        colors = ['green', 'orange', 'blue', 'red']
        ax6.bar(categories, values, color=colors, alpha=0.7)
        ax6.set_title('Final Performance Summary', fontweight='bold')
        ax6.set_ylabel('Value')
        
        plt.tight_layout()
        
        # Save the graph
        os.makedirs('saved_models', exist_ok=True)
        plt.savefig('saved_models/training_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        
        print("\n📊 Professional training graphs generated!")
        plt.show()

    def save_training_history(self):
        """Save detailed training history to JSON"""
        history_with_stats = {
            **self.training_history,
            'training_summary': {
                'total_epochs': len(self.training_history['train_loss']),
                'best_train_accuracy': max(self.training_history['train_accuracy']) if self.training_history['train_accuracy'] else 0,
                'best_val_accuracy': max(self.training_history['val_accuracy']) if self.training_history['val_accuracy'] else 0,
                'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 0,
                'final_val_loss': self.training_history['val_loss'][-1] if max(self.training_history['val_loss']) > 0 else 0,
                'total_training_time': sum(self.training_history['epoch_times']),
                'average_epoch_time': np.mean(self.training_history['epoch_times']) if self.training_history['epoch_times'] else 0
            }
        }
        
        with open('training_history.json', 'w') as f:
            json.dump(history_with_stats, f, indent=2)
        
        os.makedirs('saved_models', exist_ok=True)
        with open('saved_models/training_history.json', 'w') as f:
            json.dump(history_with_stats, f, indent=2)

    def print_training_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*70)
        print("🧬 DNA ENCRYPTION LLM TRAINING SUMMARY")
        print("="*70)
        
        if self.training_history['train_accuracy']:
            print(f"🎯 Best Training Accuracy: {max(self.training_history['train_accuracy']):.2f}%")
        if self.training_history['val_accuracy'] and max(self.training_history['val_accuracy']) > 0:
            print(f"✅ Best Validation Accuracy: {max(self.training_history['val_accuracy']):.2f}%")
        if self.training_history['train_loss']:
            print(f"📉 Final Training Loss: {self.training_history['train_loss'][-1]:.4f}")
        if max(self.training_history['val_loss']) > 0:
            print(f"📊 Final Validation Loss: {self.training_history['val_loss'][-1]:.4f}")
        
        total_time = sum(self.training_history['epoch_times'])
        print(f"⏱️ Total Training Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"📈 Average Epoch Time: {np.mean(self.training_history['epoch_times']):.2f}s")
        print("="*70)

    def save_checkpoint(self, filepath):
        """Save comprehensive checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer': self.tokenizer,
            'config': self.config,
            'training_history': self.training_history,
            'epoch': len(self.training_history['train_loss'])
        }, filepath)
        print(f"💾 Checkpoint saved: {filepath}")

    def save_model(self, filepath):
        """Save final trained model with complete history"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'model_config': self.config.model_config,
            'config_type': self.config.config_type,
            'training_history': self.training_history
        }, filepath)
        print(f"🎉 Final model saved: {filepath}")
