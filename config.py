import torch
import psutil

class Config:
    """Configuration settings based on hardware detection"""
    
    def __init__(self, config_type='auto'):
        # Hardware detection
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        self.has_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.has_gpu else 'cpu')
        
        # Auto-detect configuration
        if config_type == 'auto':
            if self.ram_gb < 8:
                config_type = 'minimal'
            elif self.ram_gb < 16:
                config_type = 'standard'
            else:
                config_type = 'large'
        
        self._set_config(config_type)
    
    def _set_config(self, config_type):
        """Set configuration based on type"""
        
        if config_type == 'minimal':
            # For laptops with 4-8GB RAM
            self.model_config = {
                'd_model': 64,
                'nhead': 4,
                'num_encoder_layers': 2,
                'num_decoder_layers': 2,
                'dim_feedforward': 256,
                'max_text_len': 64,
                'max_dna_len': 128
            }
            self.training_config = {
                'batch_size': 4,
                'epochs': 3,
                'learning_rate': 0.001,
                'num_samples': 500
            }
            
        elif config_type == 'standard':
            # For laptops with 8-16GB RAM
            self.model_config = {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'num_decoder_layers': 3,
                'dim_feedforward': 512,
                'max_text_len': 128,
                'max_dna_len': 256
            }
            self.training_config = {
                'batch_size': 8,
                'epochs': 5,
                'learning_rate': 0.0005,
                'num_samples': 2000
            }
            
        else:  # 'large'
            # For powerful laptops with 16GB+ RAM
            self.model_config = {
                'd_model': 256,
                'nhead': 8,
                'num_encoder_layers': 4,
                'num_decoder_layers': 4,
                'dim_feedforward': 1024,
                'max_text_len': 128,
                'max_dna_len': 256
            }
            self.training_config = {
                'batch_size': 16,
                'epochs': 10,
                'learning_rate': 0.0005,
                'num_samples': 5000
            }
        
        self.config_type = config_type
        print(f"Configuration: {config_type.upper()}")
        print(f"Device: {self.device}")
        print(f"Model parameters: ~{self._estimate_params():.1f}M")
    
    def _estimate_params(self):
        """Estimate model parameters in millions"""
        d = self.model_config['d_model']
        layers = self.model_config['num_encoder_layers'] + self.model_config['num_decoder_layers']
        # Rough estimation
        return (d * d * 4 * layers + d * 200) / 1e6