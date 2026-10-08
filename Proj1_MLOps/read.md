# --- source
https://colab.research.google.com/drive/15WTn5j9gk4_DHu6HQCnRqllPqVzLiR6B#scrollTo=0EZ-5MR5KWjg
https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset

# --- setup
cd Proj1_MLOps
python -m venv .venv
.venv\Scripts\activate

# --- install

pip install onnx onnxruntime
pip install fastapi uvicorn python-multipart numpy Pillow

# --- run
cd Proj1_MLOps

uvicorn main:app --reload

# --- deploy

# --- frontend
cd Proj1_MLOps
cd frontend

- setup
npx create-next-app@latest frontend

- run
npm run dev