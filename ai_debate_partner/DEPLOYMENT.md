# Deployment Notes (Vercel + Streamlit)

## Important
This project uses Streamlit as an interactive Python web app. Streamlit requires a long-running Python process and websocket-style interaction, while Vercel is optimized for serverless request/response functions.

Because of this, deploying the full Streamlit app directly on Vercel is not reliable for production.

## Recommended Deployment (Best for This Project)
Use **Streamlit Community Cloud** or **Render/Railway** for the full app.

### Option A: Streamlit Community Cloud (easiest)
1. Push this project to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from your repo.
4. Set main file to `ai_debate_partner/app.py`.
5. Add secret `OPENAI_API_KEY` in Streamlit Secrets.
6. Deploy.

### Option B: Render/Railway (full Python hosting)
1. Create a Web Service from your repo.
2. Build command:
   `pip install -r requirements.txt`
3. Start command:
   `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
4. Set root directory to `ai_debate_partner`.
5. Add env variable `OPENAI_API_KEY`.
6. Deploy.

## If Your Course Requires Vercel Specifically
Use a hybrid model:
- Deploy the Streamlit app on Render/Railway/Streamlit Cloud.
- Deploy a static landing page on Vercel that links to the live Streamlit URL.

This still gives you a Vercel deployment while keeping the app functional.

## Why this is academically valid
Your key objectives are still fully deployed and testable:
- OpenAI Speech-to-Text
- Fuzzy logic scoring
- Debate feedback and counter-argument generation
- Streamlit UI with transcript verification

The hosting platform does not change the core research/engineering contribution.
