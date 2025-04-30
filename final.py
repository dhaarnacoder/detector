import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array

# Load models
age_model = load_model('models_directory/age_model.h5')
gender_model = load_model('models_directory/gender_model.h5')
emotion_model = load_model('models_directory/emotion_model.h5')

# Labels
age_labels = ['Infant', 'Toddler', 'Child', 'Teen', 'Adult', 'Middle_Age_Adult', 'Senior_Adult']
gender_labels = ['Male', 'Female']
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start webcam
cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grayscale, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = grayscale[y:y+h, x:x+w]

        # Resize to 48x48 for all models
        resized_face = cv2.resize(roi_gray, (48, 48))
        input_array = img_to_array(resized_face) / 255.0
        input_array = np.expand_dims(input_array, axis=0)
        input_array = np.expand_dims(input_array, axis=-1)

        # Predict
        age_pred = age_model.predict(input_array)
        gender_pred = gender_model.predict(input_array)
        emotion_pred = emotion_model.predict(input_array)

        age_label = age_labels[np.argmax(age_pred)]
        gender_label = gender_labels[np.argmax(gender_pred)]
        emotion_label = emotion_labels[np.argmax(emotion_pred)]

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label = f"{gender_label}, {age_label}, {emotion_label}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Age, Gender & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
