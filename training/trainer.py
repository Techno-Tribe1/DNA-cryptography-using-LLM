import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

class DNATrainer:
    """Training pipeline"""
    
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
        
        print(f"Trainer initialized on {self.device}")
    
    def train(self, train_texts):
        """Main training loop"""
        from training.dataset import DNACryptoDataset
        
        # Create dataset
        dataset = DNACryptoDataset(
            train_texts, 
            self.tokenizer,
            self.config.model_config['max_text_len'],
            self.config.model_config['max_dna_len']
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.training_config['batch_size'],
            shuffle=True
        )
        
        self.model.train()
        
        for epoch in range(self.config.training_config['epochs']):
            self._train_epoch(epoch, dataloader)
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(f"saved_models/checkpoint_epoch_{epoch+1}.pth")
        
        # Save final model
        self.save_model("saved_models/final_model.pth")
    
    def _train_epoch(self, epoch, dataloader):
        """Train single epoch"""
        total_loss = 0
        encrypt_losses = []
        decrypt_losses = []
        
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
            
            # Encryption loss
            encrypt_output = self.model.encrypt_forward(
                text_tokens, dna_input, 
                self.tokenizer.pad_id, self.tokenizer.dna_pad_id
            )
            loss_encrypt = self.criterion(
                encrypt_output.reshape(-1, encrypt_output.size(-1)),
                dna_target.reshape(-1)
            )
            
            # Decryption loss
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
            
            # Track losses
            total_loss += total_batch_loss.item()
            encrypt_losses.append(loss_encrypt.item())
            decrypt_losses.append(loss_decrypt.item())
            
            # Progress reporting
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.training_config['epochs']}, "
                      f"Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {total_batch_loss.item():.4f}")
        
        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        avg_encrypt = sum(encrypt_losses) / len(encrypt_losses)
        avg_decrypt = sum(decrypt_losses) / len(decrypt_losses)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Encrypt Loss: {avg_encrypt:.4f}")
        print(f"  Decrypt Loss: {avg_decrypt:.4f}")
        print("-" * 50)
    
    def save_checkpoint(self, filepath):
        """Save training checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer': self.tokenizer,
            'config': self.config
        }, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def save_model(self, filepath):
        """Save final trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'model_config': self.config.model_config,
            'config_type': self.config.config_type
        }, filepath)
        print(f"Final model saved: {filepath}")