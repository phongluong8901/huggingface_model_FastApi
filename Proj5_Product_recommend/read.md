https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

# --- setup
cd Proj5_Product_recommend

python -m venv .venv
.venv\Scripts\activate

# --- install
pip install flask pandas numpy scikit-learn sentence-transformers


# --- run
cd Proj5_Product_recommend

python main.py

# --- deploy

# --- frontend
cd Proj5_Product_recommend



