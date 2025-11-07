# Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub account
- Streamlit Community Cloud account (free at https://share.streamlit.io)
- API keys ready:
  - Anthropic API key
  - OpenAI API key
  - Pinecone API key

## Step 1: Prepare Your Repository

### 1.1 Check required files
Ensure these files are in your repo:
- ✅ `app.py` (main Streamlit app)
- ✅ `chatbot.py` (RAG logic)
- ✅ `requirements.txt` (dependencies)
- ✅ `lipstick_qa_system_message (1).txt` (system prompt)
- ✅ `.gitignore` (excludes secrets)

### 1.2 Verify .gitignore
Make sure `.gitignore` includes:
```
.env
.env.*
*.env
.streamlit/secrets.toml
```

**⚠️ IMPORTANT:** Do NOT commit `.env` or `.streamlit/secrets.toml`

## Step 2: Push to GitHub

### 2.1 Initialize git (if not already done)
```bash
cd /home/sid/Documents/QnA_Chatbot
git init
git add .
git commit -m "Initial commit: Lipstick QnA chatbot"
```

### 2.2 Create GitHub repository
1. Go to https://github.com/new
2. Name: `lipstick-qna-chatbot` (or your choice)
3. Visibility: Public or Private
4. Do NOT initialize with README (you already have files)
5. Click "Create repository"

### 2.3 Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/lipstick-qna-chatbot.git
git branch -M main
git push -u origin main
```

## Step 3: Deploy on Streamlit Cloud

### 3.1 Create new app
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - **Repository:** YOUR_USERNAME/lipstick-qna-chatbot
   - **Branch:** main
   - **Main file path:** app.py
5. Click "Advanced settings"

### 3.2 Configure secrets
In the "Secrets" section, paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_MAX_TOKENS = "2000"
ANTHROPIC_TEMPERATURE = "0.7"

OPENAI_API_KEY = "sk-..."
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

PINECONE_API_KEY = "pcsk_..."
PINECONE_INDEX_NAME = "qna-reports-lipstick-saas"
PINECONE_NAMESPACE = "default"
PINECONE_TOP_K = "1"

SYSTEM_PROMPT_PATH = "lipstick_qa_system_message (1).txt"
HISTORY_MAX_TURNS = "5"
SHOW_DEBUG = "0"
```

**Replace with your actual API keys**

### 3.3 Deploy
1. Click "Deploy"
2. Wait 2-5 minutes for build
3. Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

## Step 4: Test Your Deployment

### 4.1 Basic functionality test
1. Open your app URL
2. Ask: "How does Milani 120 Can't Even look on Indian skin tone?"
3. Follow up: "I have fair skin tone"
4. Verify:
   - ✅ Context carryover works (no need to repeat product name)
   - ✅ Clean UI (no backend details unless debug is on)
   - ✅ Session history persists across questions

### 4.2 Multi-user test
1. Open app in incognito/private window
2. Each window should have separate session history
3. Verify no history mixing

## Step 5: Optional Configurations

### 5.1 Increase retrieval candidates
For better context:
```toml
PINECONE_TOP_K = "3"  # or 5
```

### 5.2 Enable debug mode (for troubleshooting)
```toml
SHOW_DEBUG = "1"
```
Users can still toggle it off in the sidebar.

### 5.3 Custom domain (optional)
Streamlit Cloud apps get free subdomain, but you can:
- Upgrade to add custom domain
- Or use the default: `your-app-name.streamlit.app`

## Troubleshooting

### Build fails with "ModuleNotFoundError"
- Check `requirements.txt` includes all dependencies
- Ensure versions are compatible

### "API key not found" error
- Verify secrets are set in Streamlit Cloud settings
- Check for typos in key names (case-sensitive)
- Ensure no extra quotes or spaces

### Slow embeddings/retrieval
- OpenAI embeddings can take 5-15s for first call
- Consider caching or using a smaller model for dev

### App shows "Streamlit is running..."
- Wait 2-5 minutes for initial deployment
- Check build logs for errors

## Updating Your App

### After code changes:
```bash
git add .
git commit -m "Your change description"
git push
```
Streamlit Cloud auto-deploys on push to main branch.

### After secrets changes:
1. Go to app settings → Secrets
2. Update the values
3. Click "Save"
4. App will restart automatically

## Security Best Practices

✅ **DO:**
- Store all API keys in Streamlit Secrets
- Use `.gitignore` to exclude `.env` and secrets files
- Review what's committed before pushing
- Rotate API keys if accidentally committed

❌ **DON'T:**
- Hardcode API keys in code
- Commit `.env` or `.streamlit/secrets.toml`
- Share secrets publicly
- Use production keys in public repos

## Support

- Streamlit docs: https://docs.streamlit.io
- Community forum: https://discuss.streamlit.io
- Your app logs: Streamlit Cloud → App → Manage app → Logs
