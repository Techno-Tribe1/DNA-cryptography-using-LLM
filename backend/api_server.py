"""
Flask API Server for DNA Cryptography System
"""

from flask import Flask, request, jsonify
from main import SimpleDNACrypto

app = Flask(__name__)

# Manual CORS — no flask-cors, no double headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        from flask import make_response
        response = make_response('', 200)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
        return response

# Initialize the crypto system once
crypto = SimpleDNACrypto()

@app.route('/')
def index():
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

@app.route('/api/encrypt', methods=['POST', 'OPTIONS'])
def encrypt_text():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        result = crypto.encrypt(text)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/decrypt', methods=['POST', 'OPTIONS'])
def decrypt_binary():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        data = request.get_json()
        binary_key = data.get('binary_key', '')
        if not binary_key:
            return jsonify({'success': False, 'error': 'Binary key is required'}), 400
        result = crypto.decrypt(binary_key)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/supported-chars', methods=['GET', 'OPTIONS'])
def get_supported_chars():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dna-mapping', methods=['GET', 'OPTIONS'])
def get_dna_mapping():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        limit = request.args.get('limit', 10, type=int)
        char_mappings = []
        for i, (char, dna) in enumerate(list(crypto.char_to_dna.items())[:limit]):
            binary = ''.join(crypto.dna_to_binary[n] for n in dna)
            char_mappings.append({
                'character': char,
                'dna_triplet': dna,
                'binary': binary
            })
        return jsonify({
            'success': True,
            'data': {
                'char_mappings': char_mappings,
                'dna_binary_mapping': crypto.dna_to_binary,
                'total_mappings': len(crypto.char_to_dna)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)