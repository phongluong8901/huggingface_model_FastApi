https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

# --- setup
cd Proj6_Intent_Predict

python -m venv .venv
.venv\Scripts\activate

# --- install
pip install streamlit transformers torch nltk
pip install torchvision

# --- run
cd Proj6_Intent_Predict

streamlit run app.py

# --- deploy

# --- frontend
cd Proj6_Intent_Predict



