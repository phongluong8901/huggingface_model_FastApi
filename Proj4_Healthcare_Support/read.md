https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

# --- setup
cd Proj4_Healthcare_Support

python -m venv .venv
.venv\Scripts\activate

# --- install
pip install flask transformers torch pandas scikit-learn
pip install accelerate
# --- run
cd Proj4_Healthcare_Support

python main.py

# --- deploy

# --- frontend
cd Proj4_Healthcare_Support

- setup
npx create-next-app@latest frontend

- run
npm run dev