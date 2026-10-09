import re
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#app
app = Flask(__name__)

# Load model và tokenizer trực tiếp từ Hugging Face Hub
model_name = "phong8901/my-t5-domain-chatbot"
print("Đang tải model từ Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

device = model.device

# Clean the text by removing unwanted characters
def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
    text = text.strip().lower()  # Strip and convert to lower case
    return text

# Chatbot function
def chatbot(dialogue):
    dialogue = clean_text(dialogue)  # Assuming clean_text is defined
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=250)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        inputs["input_ids"],
        max_length=250,
        num_beams=4,
        early_stopping=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
    
#routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=['POST'])
def chat():
    user_mesasge = request.json.get("message", "")

    if not user_mesasge:
        return jsonify({"error": "Message id required"}), 400

    response = chatbot(user_mesasge)
    return jsonify({'response': response})


#python main
if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,debug=True)