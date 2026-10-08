https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

# --- setup
cd Proj3_Summarization

python -m venv .venv
.venv\Scripts\activate

# --- install
pip install fastapi uvicorn transformers sentencepiece jinja2
pip install torch

# --- run
cd Proj3_Summarization

uvicorn app:app --reload

# --- deploy

# --- frontend
cd Proj3_Summarization

- setup
npx create-next-app@latest frontend

- run
npm run dev