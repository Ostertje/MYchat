from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

# Load the model and tokenizer
model_name = "microsoft/phi-1_5"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a route for model inference
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['input_text']
    
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted label
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = model.config.id2label[predicted_class]
    
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
