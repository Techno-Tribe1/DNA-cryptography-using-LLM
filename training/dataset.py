from torch.utils.data import Dataset
import random
import string

class DNACryptoDataset(Dataset):
    """Dataset generator for training"""
    
    def __init__(self, texts, tokenizer, max_text_len, max_dna_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_dna_len = max_dna_len
        
        # Generate corresponding DNA sequences
        self.dna_sequences = self._generate_synthetic_dna()
    
    def _generate_synthetic_dna(self):
        """Generate deterministic DNA sequences for training"""
        sequences = []
        for text in self.texts:
            # Create DNA sequence based on text hash for consistency
            text_hash = abs(hash(text))
            random.seed(text_hash)
            
            # Length proportional to text
            dna_length = min(len(text) * 2 + 10, self.max_dna_len - 10)
            dna_seq = ''.join(random.choices(['A', 'T', 'G', 'C'], k=dna_length))
            sequences.append(dna_seq)
        
        return sequences
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        dna_seq = self.dna_sequences[idx]
        
        # Tokenize
        text_tokens = self.tokenizer.encode_text(text, self.max_text_len)
        dna_tokens = self.tokenizer.encode_dna(dna_seq, self.max_dna_len)
        
        return {
            'text_tokens': text_tokens,
            'dna_tokens': dna_tokens,
            'text': text,
            'dna_sequence': dna_seq
        }

def generate_training_data(num_samples: int) -> list:
    """Generate diverse training texts"""
    texts = []
    
    # Base phrases
    base_phrases = [
        "Hello world", "Secret message", "Password protected",
        "Confidential data", "Encrypted file", "Secure transfer",
        "Authentication token", "Private key", "Digital signature"
    ]
    
    # Common words for sentence generation
    words = [
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "up", "about", "into", "through", "during",
        "data", "file", "system", "user", "admin", "server", "network",
        "secure", "encrypt", "decode", "password", "key", "token",
        "message", "email", "document", "report", "database", "backup"
    ]
    
    # Generate samples
    for i in range(num_samples):
        if i < len(base_phrases):
            texts.append(base_phrases[i])
        else:
            # Random sentences
            length = random.randint(2, 10)
            sentence = ' '.join(random.choices(words, k=length))
            texts.append(sentence.capitalize())
    
    return texts