from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.image_classifier import classify_image
from models.text_classifier import classify_text

app = Flask(__name__)

# Route for image classification (handles one image at a time)
@app.route('/classify_image', methods=['POST'])
def classify_image_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file:
        result = classify_image(file)
        return jsonify(result), 200
    else:
        return jsonify({'error': 'Invalid image file'}), 400

# Route for text classification (remains unchanged)
@app.route('/classify_text', methods=['POST'])
def classify_text_route():
    data = request.json
    if 'description' not in data:
        return jsonify({'error': 'No text provided'}), 400

    description = data['description']
    result = classify_text(description)
    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True)