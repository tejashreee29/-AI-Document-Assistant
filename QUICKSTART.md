# Quick Start Guide - AI Document Assistant

## 🚀 Get Started in 3 Steps

### Step 1: Install & Setup (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
touch .env
```

**Choose your AI model:**

**Option A: OpenAI (Cloud, requires API key)**
```bash
# Add to .env file:
OPENAI_API_KEY=sk-your-api-key-here
MODEL_TYPE=openai
MODEL_NAME=gpt-3.5-turbo
```

**Option B: Ollama (Free, Local, NO API key needed)** ⭐ Recommended
```bash
# Install Ollama
brew install ollama

# Start Ollama
ollama serve

# Download a model (in another terminal)
ollama pull llama2

# Add to .env file:
MODEL_TYPE=ollama
MODEL_NAME=llama2
```

### Step 2: Run the App

```bash
streamlit run app.py
```

The app will open at: http://localhost:8501

### Step 3: Use the App

1. **Login/Register** - Create an account
2. **Upload PDFs** - Click "Browse files" in sidebar
3. **Process** - Click "Process Documents" (wait for completion)
4. **Ask Questions** - Start chatting with your documents!

---

## 📊 What to Expect

### Processing Times
- **Upload**: 1-5 seconds per PDF
- **Extraction**: 2-10 seconds per PDF  
- **Embedding**: 5-30 seconds total
- **Queries**: 10-30 seconds per question

### You'll See:
- ✅ Progress bars showing completion
- ✅ Status messages at each stage
- ✅ Word/character count after extraction
- ✅ "Searching documents..." when querying
- ✅ "Generating answer..." during AI processing

---

## 💡 Tips for Best Results

### Document Upload
- ✅ Use PDFs with selectable text (not scanned images)
- ✅ Keep PDFs under 10MB each
- ✅ Upload 1-5 documents at a time
- ✅ Ensure PDFs are not password-protected

### Asking Questions
- ✅ Be specific: "What was the revenue in Q3 2023?"
- ❌ Too vague: "What about revenue?"
- ✅ Use keywords from your documents
- ✅ Ask one question at a time
- ✅ Keep questions under 200 characters

### Model Selection
- **Speed**: gpt-3.5-turbo (OpenAI) or llama2 (Ollama)
- **Quality**: gpt-4 (OpenAI) or mistral (Ollama)
- **Free**: Ollama (any model)
- **Cost**: OpenAI (pay per use)

---

## 🔥 Quick Fixes

### "No text extracted"
→ Your PDF has scanned images. Use PDFs with selectable text.

### "API Quota Exceeded"
→ Switch to Ollama (free, local) in the sidebar.

### "Cannot connect to Ollama"
→ Run `ollama serve` in Terminal.

### "Taking too long"
→ Use smaller PDFs or ask simpler questions.

### "No relevant information"
→ Rephrase your question with keywords from your documents.

---

## 🎯 Example Workflow

```
1. Start app: streamlit run app.py
2. Login with username/password
3. Upload: research_paper.pdf
4. Click: "Process Documents"
5. Wait: ~20 seconds (you'll see progress)
6. Ask: "What is the main conclusion of this research?"
7. Get answer with source citations!
```

---

## 📱 Interface Overview

### Sidebar (Left)
- 👤 User Profile & Logout
- ⚙️ Model Settings (OpenAI/Ollama/HuggingFace)
- 📁 Document Upload
- 💬 Chat Sessions
- 🔑 API Status

### Main Area (Center)
- 💬 Chat interface
- 📄 Source documents (expandable)
- 🎯 Welcome screen (before upload)

### Features
- 🧠 Conversation memory (remembers context)
- 📄 Source highlighting (see where answers come from)
- 💾 Auto-save (all chats saved to database)
- 🔄 Session management (switch between conversations)

---

## 🆘 Need Help?

1. **Check error messages** - They now include step-by-step fixes
2. **See TROUBLESHOOTING.md** - Comprehensive guide for all issues
3. **Try Ollama** - Free, local, no API key needed
4. **Check sidebar** - Shows API status and model info

---

## 🎉 You're Ready!

Your AI Document Assistant now has:
- ⚡ Fast, responsive processing
- 🎯 Clear progress indicators  
- 💡 Helpful error messages
- ✅ Input validation
- 📊 Processing statistics

**Start asking questions about your documents!** 🚀
