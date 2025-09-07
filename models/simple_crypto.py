
import random
import string
import pickle

class SimpleDNACrypto:
    """Simple deterministic DNA encryption for immediate use"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        
        # Character set
        self.chars = list(string.printable)
        self.dna_bases = ['A', 'T', 'G', 'C']
        
        # Create bijective mapping
        self.char_to_dna = {}
        self.dna_to_char = {}
        
        used_triplets = set()
        for char in self.chars:
            while True:
                triplet = ''.join(random.choices(self.dna_bases, k=3))
                if triplet not in used_triplets:
                    used_triplets.add(triplet)
                    self.char_to_dna[char] = triplet
                    self.dna_to_char[triplet] = char
                    break
    
    def encrypt(self, text: str) -> dict:
        """Encrypt text to DNA and binary"""
        dna_sequence = ''.join(self.char_to_dna.get(char, 'AAA') for char in text)
        binary_data = self.dna_to_binary(dna_sequence)
        
        return {
            'original': text,
            'dna_sequence': dna_sequence,
            'binary_data': binary_data,
            'stats': {
                'original_length': len(text),
                'dna_length': len(dna_sequence),
                'binary_length': len(binary_data),
                'compression_ratio': len(binary_data) / (len(text) * 8) if text else 0
            }
        }
    
    def decrypt(self, dna_sequence=None, binary_data=None) -> str:
        """Decrypt from DNA or binary"""
        if binary_data and not dna_sequence:
            dna_sequence = self.binary_to_dna(binary_data)
        
        text = ""
        for i in range(0, len(dna_sequence), 3):
            triplet = dna_sequence[i:i+3]
            if len(triplet) == 3 and triplet in self.dna_to_char:
                text += self.dna_to_char[triplet]
        
        return text
    
    def dna_to_binary(self, dna_sequence: str) -> str:
        """DNA to binary conversion"""
        mapping = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
        return ''.join(mapping.get(base, '00') for base in dna_sequence)
    
    def binary_to_dna(self, binary_data: str) -> str:
        """Binary to DNA conversion"""
        mapping = {'00': 'A', '01': 'T', '10': 'G', '11': 'C'}
        dna = ""
        for i in range(0, len(binary_data), 2):
            if i + 1 < len(binary_data):
                pair = binary_data[i:i+2]
                dna += mapping.get(pair, 'A')
        return dna
    
    def save(self, filepath: str):
        """Save the crypto system"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load saved crypto system"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)