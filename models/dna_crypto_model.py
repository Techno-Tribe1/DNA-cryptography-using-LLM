import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DNACryptoModel(nn.Module):
    """Main DNA encryption transformer model"""
    
    def __init__(self, text_vocab_size, dna_vocab_size, **model_config):
        super().__init__()
        
        # Extract config
        d_model = model_config.get('d_model', 128)
        nhead = model_config.get('nhead', 8)
        num_encoder_layers = model_config.get('num_encoder_layers', 3)
        num_decoder_layers = model_config.get('num_decoder_layers', 3)
        dim_feedforward = model_config.get('dim_feedforward', 512)
        max_text_len = model_config.get('max_text_len', 128)
        max_dna_len = model_config.get('max_dna_len', 256)
        
        self.d_model = d_model
        self.max_text_len = max_text_len
        self.max_dna_len = max_dna_len
        
        # Embeddings
        self.text_embedding = nn.Embedding(text_vocab_size, d_model)
        self.dna_embedding = nn.Embedding(dna_vocab_size, d_model)
        
        # Positional encodings
        self.text_pos_encoding = PositionalEncoding(d_model, max_text_len)
        self.dna_pos_encoding = PositionalEncoding(d_model, max_dna_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.dna_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Reverse path for decryption
        self.dna_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projections
        self.dna_output = nn.Linear(d_model, dna_vocab_size)
        self.text_output = nn.Linear(d_model, text_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_masks(self, src_tokens, tgt_tokens, src_pad_id, tgt_pad_id):
        """Create attention masks"""
        src_mask = (src_tokens == src_pad_id)
        tgt_mask = (tgt_tokens == tgt_pad_id)
        
        # Causal mask for target sequence
        seq_len = tgt_tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(tgt_tokens.device)
        
        return src_mask, tgt_mask, causal_mask
    
    def encrypt_forward(self, text_tokens, dna_tokens, text_pad_id, dna_pad_id):
        """Forward pass for encryption (text -> DNA)"""
        # Create masks
        text_mask, dna_mask, causal_mask = self.create_masks(
            text_tokens, dna_tokens, text_pad_id, dna_pad_id
        )
        
        # Encode text
        text_embed = self.text_embedding(text_tokens) * math.sqrt(self.d_model)
        text_embed = self.text_pos_encoding(text_embed.transpose(0, 1)).transpose(0, 1)
        memory = self.text_encoder(text_embed, src_key_padding_mask=text_mask)
        
        # Decode to DNA
        dna_embed = self.dna_embedding(dna_tokens) * math.sqrt(self.d_model)
        dna_embed = self.dna_pos_encoding(dna_embed.transpose(0, 1)).transpose(0, 1)
        
        dna_output = self.dna_decoder(
            dna_embed, memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=text_mask,
            tgt_key_padding_mask=dna_mask
        )
        
        return self.dna_output(dna_output)
    
    def decrypt_forward(self, dna_tokens, text_tokens, dna_pad_id, text_pad_id):
        """Forward pass for decryption (DNA -> text)"""
        # Create masks
        dna_mask, text_mask, causal_mask = self.create_masks(
            dna_tokens, text_tokens, dna_pad_id, text_pad_id
        )
        
        # Encode DNA
        dna_embed = self.dna_embedding(dna_tokens) * math.sqrt(self.d_model)
        dna_embed = self.dna_pos_encoding(dna_embed.transpose(0, 1)).transpose(0, 1)
        memory = self.dna_encoder(dna_embed, src_key_padding_mask=dna_mask)
        
        # Decode to text
        text_embed = self.text_embedding(text_tokens) * math.sqrt(self.d_model)
        text_embed = self.text_pos_encoding(text_embed.transpose(0, 1)).transpose(0, 1)
        
        text_output = self.text_decoder(
            text_embed, memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=dna_mask,
            tgt_key_padding_mask=text_mask
        )
        
        return self.text_output(text_output)