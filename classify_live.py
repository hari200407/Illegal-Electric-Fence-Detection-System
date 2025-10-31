import cv2
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from collections import deque

# Keep last 10 predictions
prediction_history = deque(maxlen=10)

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

video_path = 0
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video/webcam")
    exit()

print("✅ Video/Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video ended")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    x = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(probs.argmax())
    fence_type = class_names[top_idx]
    confidence = float(probs[top_idx])

    if confidence > 0.8:
        prediction_history.append(fence_type)
    else:
        prediction_history.append("fence")

    most_common = max(set(prediction_history), key=prediction_history.count)
    label = f"{most_common} ({confidence:.2f})"

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Fence Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
