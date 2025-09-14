# Create fixed_simple_crypto.py (from my earlier solution)
import random
import string

class FixedDNACrypto:
    def __init__(self, seed=42):
        random.seed(seed)
        # Use only 64 characters to match 4³ DNA triplets
        self.chars = list(string.ascii_letters + string.digits + " .,!?-")[:64]
        
        # Generate all possible triplets
        all_triplets = []
        for b1 in ['A', 'T', 'G', 'C']:
            for b2 in ['A', 'T', 'G', 'C']:
                for b3 in ['A', 'T', 'G', 'C']:
                    all_triplets.append(b1 + b2 + b3)
        
        random.shuffle(all_triplets)
        
        # Create bijective mapping
        self.char_to_dna = {}
        self.dna_to_char = {}
        
        for i, char in enumerate(self.chars):
            triplet = all_triplets[i]
            self.char_to_dna[char] = triplet
            self.dna_to_char[triplet] = char
    
    def encrypt(self, text):
        # Handle unsupported characters
        filtered = ''.join(c if c in self.char_to_dna else '?' for c in text)
        dna = ''.join(self.char_to_dna.get(c, 'AAA') for c in filtered)
        return {
            'original': text,
            'dna_sequence': dna,
            'binary_data': self._dna_to_binary(dna)
        }
    
    def decrypt(self, dna_sequence):
        text = ""
        for i in range(0, len(dna_sequence), 3):
            triplet = dna_sequence[i:i+3]
            if len(triplet) == 3:
                text += self.dna_to_char.get(triplet, '?')
        return text
    
    def _dna_to_binary(self, dna):
        mapping = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
        return ''.join(mapping.get(base, '00') for base in dna)

# Test it
crypto = FixedDNACrypto()
result = crypto.encrypt("Hello World!")
decrypted = crypto.decrypt(result['dna_sequence'])
print(f"'{result['original']}' → '{decrypted}' | Match: {result['original'] == decrypted}")
