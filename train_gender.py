import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set up paths
base_dir = r'D:\AI(Scanner)\emotion_age_gender_detector\dataset\utkface_aligned_cropped\Gender'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)

# Model architecture
gender_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes: Male and Female
])

# Compile the model
gender_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
gender_model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# Save the model
model_dir = r'D:\AI(Scanner)\emotion_age_gender_detector\models_directory'
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'gender_model.h5')
gender_model.save(model_path)

print(f"✅ Gender model saved at: {model_path}")
