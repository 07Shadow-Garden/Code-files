from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load the trained models
banana_model = load_model("banana_disease_detection_model.keras")
mango_model = load_model("mango_disease_detection_model.keras")

# Class label mappings
banana_classes = {
    0: "Black Sigatoka",
    1: "Bract Mosaic Virus",
    2: "Insect Pest",
    3: "Moko",
    4: "Panama",
    5: "Yellow Sigatoka",
    6: "Banana Healthy Leaf"
}

mango_classes = {
    0: "Anthracnose",
    1: "Bacterial Canker",
    2: "Cutting Weevil",
    3: "Die Black",
    4: "Gall Midge",
    5: "Powdery Mildew",
    6: "Sooty Mould",
    7: "Mango Healthy"
}


def prepare_image(image_path, target_size=(150, 150)):
    """Preprocess the image for prediction."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if "model_type" not in request.form:
        return jsonify({"error": "No model type specified."}), 400

    model_type = request.form["model_type"].lower()
    if model_type not in ["banana", "mango"]:
        return jsonify({"error": "Invalid model type. Choose 'banana' or 'mango'."}), 400

    # Save the uploaded file
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Prepare the image
    image = prepare_image(file_path)

    # Select the model and class labels
    if model_type == "banana":
        model = banana_model
        class_labels = banana_classes
    else:
        model = mango_model
        class_labels = mango_classes

    # Make predictions
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    confidence_score = float(np.max(predictions[0]))

    # Remove the uploaded file
    os.remove(file_path)

    return jsonify({
        "predicted_class": class_labels[predicted_class],
        "confidence_score": confidence_score
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
