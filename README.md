# BorderComply Pakistan — Groq edition

## 1. Get the Groq key
1. Sign in at https://console.groq.com → **API Keys** → **Create API Key**.
2. Copy the key (starts with `gsk_`). It is shown only once.

## 2. Add it to Streamlit Cloud
1. Open your app on https://share.streamlit.io → **⋮ → Settings → Secrets**.
2. Delete the old `GOOGLE_API_KEY` line and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
3. **Save**. The app restarts automatically.
4. Python version: 3.11, 3.12 or 3.13 all work (CrewAI was removed, so the 3.12 lock no longer applies).

## 3. Run locally (optional)
```bash
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # then put your key in it
streamlit run app.py
```

## Model
Default: `openai/gpt-oss-20b` (Groq free tier, fastest, guaranteed-valid JSON).
To switch, add to Secrets: `GROQ_MODEL = "openai/gpt-oss-120b"` (smarter, a bit slower)
or `GROQ_MODEL = "llama-3.3-70b-versatile"`.

## Notes
- One AI call per check (~5k tokens), official sources fetched in parallel and cached for 6 h.
- Never commit `.streamlit/secrets.toml` (already in `.gitignore`).
