from flask import Flask, render_template, request
import torch
from transformers import DeiTForImageClassification, DeiTConfig
import cv2
from PIL import Image
import torchvision.transforms as transforms
import os
import uuid
import gdown   # <-- NEW

# ------------------------------------------
# Flask setup
# ------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("model", exist_ok=True)

# ------------------------------------------
# Device
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Fake', 'Real']


# ------------------------------------------
# GOOGLE DRIVE DOWNLOAD (gdown)
# ------------------------------------------
def download_from_gdrive(file_id, output_path):
    print("📥 Checking model...")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 5000:
        print("✅ Model already exists — skipping download.")
        return

    print("📥 Downloading model using gdown...")
    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, output_path, quiet=False)

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 5000:
        raise ValueError("❌ Model download failed! File is too small or corrupted.")

    print("✅ Model downloaded successfully!")


# ------------------------------------------
# Load Model
# ------------------------------------------
student_config = DeiTConfig(
    image_size=112,
    num_labels=2,
    patch_size=16,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)

FILE_ID = "13JrLbtbp_DkrXbv4IYSQm1ZzkeGobwgg"   # <-- Your file ID
model_path = "model/best_student.pth"

# Download model
download_from_gdrive(FILE_ID, model_path)

# Load model
model = DeiTForImageClassification(student_config)

print("🔍 Loading state_dict...")
state_dict = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

print("🚀 Model loaded successfully!")


# ------------------------------------------
# Face Detection
# ------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def preprocess_image(path, target_size=(112, 112), padding=20):
    image = cv2.imread(path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + padding * 2, image.shape[1] - x)
    h = min(h + padding * 2, image.shape[0] - y)

    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, target_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face)


# ------------------------------------------
# Transform
# ------------------------------------------
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------
# Routes
# ------------------------------------------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded"

    image_file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    face_img = preprocess_image(image_path)
    if face_img is None:
        return render_template(
            "index.html",
            result="❌ No face detected",
            file_path=image_path
        )

    image_tensor = data_transforms(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs.logits[0]
        probabilities = torch.softmax(logits, dim=0)

        predicted_idx = torch.argmax(logits).item()
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx] * 100)

    return render_template(
        "index.html",
        result=f"Prediction: {predicted_class} ({confidence:.2f}%)",
        file_path=image_path
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
