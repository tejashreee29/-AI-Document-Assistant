# Troubleshooting Guide - AI Document Assistant

## 🔧 Common Issues & Solutions

### 1. "No text could be extracted from PDFs"
**Cause**: Your PDF contains scanned images, not searchable text.

**Solutions**:
- Use PDFs with selectable/copyable text
- If you have scanned PDFs, use OCR software first
- Try a different PDF to test if the issue is file-specific

---

### 2. "OpenAI API Quota Exceeded"
**Cause**: You've hit your OpenAI API usage limit.

**Solutions**:
1. **Quick Fix**: Switch to Ollama (free, local) in sidebar
2. **Check Billing**: https://platform.openai.com/account/billing
3. **Add Payment Method**: Increase your quota
4. **Wait**: Free tier resets monthly

---

### 3. "Cannot connect to Ollama server"
**Cause**: Ollama is not running or not installed.

**Solutions**:
```bash
# Install Ollama (if not installed)
brew install ollama

# Start Ollama
ollama serve

# Download a model
ollama pull llama2

# Verify it's running
ollama list
```

---

### 4. "Request Timeout / Taking Too Long"
**Cause**: Large documents or complex questions taking too long.

**Solutions**:
- Ask simpler, more specific questions
- Upload smaller PDF files
- Use faster models (gpt-3.5-turbo)
- Check your internet connection

---

### 5. "No relevant information found"
**Cause**: Question doesn't match content in documents.

**Solutions**:
- Rephrase using keywords from your documents
- Ask more specific questions
- Verify you uploaded the correct documents

---

### 6. "Invalid API Key"
**Cause**: OpenAI API key is missing or incorrect.

**Solutions**:
1. Create `.env` file: `touch .env`
2. Add: `OPENAI_API_KEY=sk-your-key-here`
3. Get key: https://platform.openai.com/api-keys
4. Restart app

**Alternative**: Use Ollama (no API key needed)

---

### 7. "Out of Memory"
**Solutions**:
- Close other applications
- Upload fewer/smaller PDFs
- Use cloud models (OpenAI) instead of local
- Restart your computer

---

### 8. Application Won't Start
**Solutions**:
```bash
# Check Python version
python3 --version

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Clear cache
rm -rf __pycache__/

# Restart
streamlit run app.py
```

---

## 🆘 Still Having Issues?

1. Check the error message - it includes troubleshooting steps
2. Look at the sidebar - check API status and model settings
3. Try Ollama - free, local, no API key needed
4. Restart the app
