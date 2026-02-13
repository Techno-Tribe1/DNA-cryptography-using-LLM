# Step-by-Step Connection Guide

## 📋 Prerequisites

1. **Python 3.7+** installed on your system
2. **Flask** and **Flask-CORS** packages installed

---

## 🔧 Step 1: Install Required Packages

Open your terminal/command prompt in the project directory and run:

```bash
pip install flask flask-cors
```

Or if you're using a virtual environment:

```bash
# Activate your virtual environment first (if you have one)
# Then install packages
pip install flask flask-cors
```

---

## 🚀 Step 2: Start the API Server

1. Open a terminal/command prompt in your project directory
2. Run the API server:

```bash
python api_server.py
```

You should see output like:
```
🚀 Starting DNA Cryptography API Server...
📡 API will be available at: http://localhost:5000
🌐 Open frontend.html in your browser to use the UI
 * Running on http://0.0.0.0:5000
```

**⚠️ Important:** Keep this terminal window open! The server must be running for the frontend to work.

---

## 🌐 Step 3: Open the Frontend

1. **Keep the API server running** (from Step 2)
2. Open `frontend.html` in your web browser:
   - **Option A:** Double-click the `frontend.html` file
   - **Option B:** Right-click → Open with → Your browser
   - **Option C:** Drag and drop `frontend.html` into your browser

3. The frontend will automatically connect to `http://localhost:5000`

---

## ✅ Step 4: Test the Connection

1. In the browser, you should see the DNA Cryptography System interface
2. Click on the **"🔒 Encrypt"** tab
3. Type some text (e.g., "Hello World")
4. Click **"🔒 Encrypt Text"**
5. You should see the encrypted result with DNA sequence and binary key

If you see results, **connection is successful!** ✅

---

## 🔍 Troubleshooting

### Problem: "Connection Error" in the browser

**Solution:**
- Make sure `api_server.py` is running (Step 2)
- Check that the server is running on `http://localhost:5000`
- Try refreshing the browser page

### Problem: "ModuleNotFoundError: No module named 'flask'"

**Solution:**
```bash
pip install flask flask-cors
```

### Problem: Port 5000 is already in use

**Solution:**
- Close other applications using port 5000
- Or modify `api_server.py` line 75 to use a different port:
  ```python
  app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
  ```
- Then update `frontend.html` line 203:
  ```javascript
  const API_BASE_URL = 'http://localhost:5001/api';  // Change 5000 to 5001
  ```

### Problem: CORS errors in browser console

**Solution:**
- Make sure `flask-cors` is installed: `pip install flask-cors`
- The `CORS(app)` line in `api_server.py` should handle this

---

## 📁 File Structure

After setup, you should have:
```
DNA-cryptography-demo/
├── main1.py              # Your original crypto class
├── api_server.py         # Flask API server (NEW)
├── frontend.html         # Web UI (NEW)
└── CONNECTION_STEPS.md   # This file (NEW)
```

---

## 🎯 How It Works

1. **Frontend (frontend.html):**
   - User enters text or binary key
   - Sends HTTP request to API server
   - Displays results

2. **API Server (api_server.py):**
   - Receives HTTP requests
   - Uses `SimpleDNACrypto` from `main1.py`
   - Returns JSON responses

3. **Backend (main1.py):**
   - Contains the actual encryption/decryption logic
   - No changes needed!

---

## 🎉 You're All Set!

Now you can:
- ✅ Encrypt text to binary keys
- ✅ Decrypt binary keys to text
- ✅ View supported characters
- ✅ See DNA mappings

Enjoy your DNA Cryptography System! 🧬🔐

