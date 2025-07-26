import os
import cv2
import torch
from flask import Flask, render_template, Response, send_file, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import gc
import datetime

app = Flask(__name__)

# Load models and processors
gender_processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")

age_processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
age_model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gender_model.to(device)
age_model.to(device)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Video capture
cap = cv2.VideoCapture(0)
output_frame = None

# Global variables for storing the latest predictions
latest_gender = "Unknown"
latest_age = "Unknown"

def preprocess_image(face, processor):
    image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)
    return inputs

def predict_gender(face):
    inputs = preprocess_image(face, gender_processor)
    with torch.no_grad():
        outputs = gender_model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    label = gender_model.config.id2label[predicted_class]
    score = torch.softmax(logits, dim=-1).max().item()
    return label, score

def predict_age(face):
    inputs = preprocess_image(face, age_processor)
    with torch.no_grad():
        outputs = age_model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    label = age_model.config.id2label[predicted_class]
    score = torch.softmax(logits, dim=-1).max().item()
    return label, score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global output_frame, latest_gender, latest_age

    def generate():
        global output_frame, latest_gender, latest_age

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (128, 128))

                try:
                    # Predict gender and age
                    gender_label, gender_score = predict_gender(resized_face)
                    age_label, age_score = predict_age(resized_face)

                    # Update the latest predictions
                    latest_gender = f"{gender_label} ({gender_score:.2f})"
                    latest_age = f"{age_label} ({age_score:.2f})"
                except RuntimeError as e:
                    print("CUDA Out of Memory:", e)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

                # Draw results on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Gender: {gender_label} ({gender_score:.2f})", 
                            (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Age: {age_label} ({age_score:.2f})", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            output_frame = frame.copy()
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_image')
def processed_image():
    """Serve the current processed image."""
    global output_frame

    if output_frame is not None:
        filename = "output.jpg"
        cv2.imwrite(filename, output_frame)
        return send_file(filename, mimetype='image/jpeg', as_attachment=False)

    return "No processed image available", 404

@app.route('/capture_image')
def capture_image():
    """Save the current processed image for download."""
    global output_frame

    if output_frame is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_{timestamp}.jpg"
        cv2.imwrite(filename, output_frame)
        return send_file(filename, mimetype='image/jpeg', as_attachment=True)

    return "No image to capture", 404

@app.route('/results')
def results():
    global latest_gender, latest_age
    return jsonify({
        "gender": latest_gender,
        "age": latest_age
    })

if __name__ == '__main__':
    app.run(debug=True)
