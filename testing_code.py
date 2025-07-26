import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms

# Define the model
class AgeGenderModel(nn.Module):
    def __init__(self):
        super(AgeGenderModel, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.gender_fc = nn.Linear(128, 1)  # Sigmoid for gender
        self.age_fc = nn.Linear(128, 1)    # Linear for age

    def forward(self, x):
        x = self.shared_conv(x)
        x = self.flatten(x)
        x = self.shared_fc(x)
        gender = torch.sigmoid(self.gender_fc(x))  # Binary classification
        age = self.age_fc(x)  # Regression
        return gender, age

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 1: Load the trained model
model = AgeGenderModel()
model.load_state_dict(torch.load('age_gender_model_gpu.pth', map_location=torch.device('cpu')))
model.eval()

# Step 2: Define the preprocessing function
def preprocess_frame(face, transform, img_size=(128, 128)):
    face = cv2.resize(face, img_size)  # Resize the face
    face = face.astype('float32') / 255.0  # Normalize to [0, 1]
    face = transform(face).unsqueeze(0)  # Add batch dimension
    return face

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Step 3: Start the webcam for real-time detection
cap = cv2.VideoCapture(0)  # Open the webcam

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]
        input_face = preprocess_frame(face, transform)

        # Make predictions
        with torch.no_grad():
            gender_pred, age_pred = model(input_face)

        # Decode predictions
        gender = "Male" if gender_pred.item() < 0.5 else "Female"
        age = int(age_pred.item() * 100)  # Scale age back to original range

        # Display predictions on the frame
        label = f"Gender: {gender}, Age: {age}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Age and Gender Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
