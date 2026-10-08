https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

# --- setup
cd Proj2_Sentiment

python -m venv .venv
.venv\Scripts\activate

# --- install
pip install -r requirements.txt

# --- run
cd Proj2_Sentiment

uvicorn app.main:app --reload

# --- deploy

# --- frontend
cd Proj2_Sentiment

- setup
npx create-next-app@latest frontend

- run
npm run dev