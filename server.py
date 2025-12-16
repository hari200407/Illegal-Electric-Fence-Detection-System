
import cv2
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("fence_classifier.pth", map_location=device)
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(checkpoint['class_names']))
model.load_state_dict(checkpoint['model_state'])
model.eval().to(device)

class_names = checkpoint['class_names']

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.route("/frames", methods=["POST"])
def process_frame():
    file = request.files['frame']
    img = Image.open(file.stream).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(probs.argmax())
    fence_type = class_names[top_idx]
    confidence = float(probs[top_idx])

    return jsonify({"fence_type": fence_type, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
