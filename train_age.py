from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# Paths
age_train_dir = r'D:\AI(Scanner)\emotion_age_gender_detector\dataset\utkface_aligned_cropped\Age\train'
age_val_dir = r'D:\AI(Scanner)\emotion_age_gender_detector\dataset\utkface_aligned_cropped\Age\validation'

# Image data generators
datagen = ImageDataGenerator(rescale=1./255)

train = datagen.flow_from_directory(age_train_dir, target_size=(48,48), color_mode='grayscale',
                                    batch_size=64, class_mode='categorical')
val = datagen.flow_from_directory(age_val_dir, target_size=(48,48), color_mode='grayscale',
                                  batch_size=64, class_mode='categorical')

# Age classification model
age_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 age groups
])

age_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train age model
age_model.fit(train, validation_data=val, epochs=15)

# Save model
age_model.save('models_directory/age_model.h5')
