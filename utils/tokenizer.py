import torch
import string
from typing import List, Dict

class DNACryptoTokenizer:
    """Enhanced tokenizer with better DNA mapping"""
    
    def __init__(self):
        # Core vocabularies
        self.dna_bases = ['A', 'T', 'G', 'C']
        self.special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        
        # Extended character set for better coverage
        self.text_chars = list(string.printable)
        
        # Create ID mappings
        all_text_tokens = self.text_chars + self.special_tokens
        all_dna_tokens = self.dna_bases + self.special_tokens
        
        self.text_to_id = {char: i for i, char in enumerate(all_text_tokens)}
        self.id_to_text = {i: char for char, i in self.text_to_id.items()}
        
        self.dna_to_id = {base: i for i, base in enumerate(all_dna_tokens)}
        self.id_to_dna = {i: base for base, i in self.dna_to_id.items()}
        
        # Vocabulary sizes
        self.text_vocab_size = len(all_text_tokens)
        self.dna_vocab_size = len(all_dna_tokens)
        
        # Special token IDs
        self.pad_id = self.text_to_id['<PAD>']
        self.start_id = self.text_to_id['<START>']
        self.end_id = self.text_to_id['<END>']
        self.unk_id = self.text_to_id['<UNK>']
        
        self.dna_pad_id = self.dna_to_id['<PAD>']
        self.dna_start_id = self.dna_to_id['<START>']
        self.dna_end_id = self.dna_to_id['<END>']
    
    def encode_text(self, text: str, max_length: int) -> torch.Tensor:
        """Encode text to tensor"""
        tokens = [self.start_id]
        for char in text:
            tokens.append(self.text_to_id.get(char, self.unk_id))
        tokens.append(self.end_id)
        
        # Pad/truncate
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.end_id]
        else:
            tokens.extend([self.pad_id] * (max_length - len(tokens)))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode_text(self, token_ids: torch.Tensor) -> str:
        """Decode tensor to text"""
        text = ""
        for token_id in token_ids:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            char = self.id_to_text.get(token_id, '<UNK>')
            if char == '<END>':
                break
            elif char not in ['<PAD>', '<START>', '<UNK>']:
                text += char
        return text
    
    def encode_dna(self, dna_sequence: str, max_length: int) -> torch.Tensor:
        """Encode DNA sequence to tensor"""
        tokens = [self.dna_start_id]
        for base in dna_sequence:
            tokens.append(self.dna_to_id.get(base, self.dna_to_id['<UNK>']))
        tokens.append(self.dna_end_id)
        
        # Pad/truncate
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.dna_end_id]
        else:
            tokens.extend([self.dna_pad_id] * (max_length - len(tokens)))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode_dna(self, token_ids: torch.Tensor) -> str:
        """Decode tensor to DNA sequence"""
        dna = ""
        for token_id in token_ids:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            base = self.id_to_dna.get(token_id, '<UNK>')
            if base == '<END>':
                break
            elif base not in ['<PAD>', '<START>', '<UNK>']:
                dna += base
        return dna
