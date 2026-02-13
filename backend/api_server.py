"""
Flask API Server for DNA Cryptography System
Connects frontend UI with main1.py SimpleDNACrypto class
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from main import SimpleDNACrypto

app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins (update with your Vercel URL after deploy)

# Initialize the crypto system once
crypto = SimpleDNACrypto()

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'DNA Cryptography API is running',
        'endpoints': {
            'encrypt': '/api/encrypt',
            'decrypt': '/api/decrypt',
            'supported_chars': '/api/supported-chars',
            'dna_mapping': '/api/dna-mapping'
        }
    })

@app.route('/api/encrypt', methods=['POST'])
def encrypt_text():
    """
    Encrypt text to compressed binary key (Base64 format)
    Returns: compressed key, DNA sequence, and compression stats
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400
        
        result = crypto.encrypt(text)
        # Result includes: binary_key (compressed), original_text, filtered_text,
        # dna_sequence, original_binary_length, compressed_length, compression_ratio, stats
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/decrypt', methods=['POST'])
def decrypt_binary():
    """
    Decrypt compressed key (Base64) or binary key to text
    Accepts: Base64 compressed keys (new format) or binary keys (old format)
    Returns: decrypted text, DNA sequence, and stats
    """
    try:
        data = request.get_json()
        binary_key = data.get('binary_key', '')
        
        if not binary_key:
            return jsonify({
                'success': False,
                'error': 'Binary key is required'
            }), 400
        
        result = crypto.decrypt(binary_key)
        # Result includes: decrypted_text, dna_sequence, binary_key, compressed_key (if applicable), stats
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/supported-chars', methods=['GET'])
def get_supported_chars():
    """Get list of supported characters"""
    try:
        chars = crypto.get_supported_chars()
        return jsonify({
            'success': True,
            'data': {
                'characters': chars,
                'count': len(chars),
                'triplets_count': len(crypto.dna_triplets)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dna-mapping', methods=['GET'])
def get_dna_mapping():
    """Get DNA mapping information"""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        # Get first N character mappings
        char_mappings = []
        for i, (char, dna) in enumerate(list(crypto.char_to_dna.items())[:limit]):
            binary = ''.join(crypto.dna_to_binary[n] for n in dna)
            char_mappings.append({
                'character': char,
                'dna_triplet': dna,
                'binary': binary
            })
        
        # DNA to binary mapping
        dna_binary_mapping = crypto.dna_to_binary
        
        return jsonify({
            'success': True,
            'data': {
                'char_mappings': char_mappings,
                'dna_binary_mapping': dna_binary_mapping,
                'total_mappings': len(crypto.char_to_dna)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("🚀 Starting DNA Cryptography API Server...")
    print(f"📡 API running at: http://0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)