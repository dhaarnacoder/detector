import os
import shutil
import random

# Base directory
base_dir = r'D:\AI(Scanner)\emotion_age_gender_detector\dataset\utkface_aligned_cropped'
source_folder = os.path.join(base_dir, 'UTKFace')

# Destination directories
age_base = os.path.join(base_dir, 'Age')
gender_base = os.path.join(base_dir, 'Gender')

# Age group mapping function
def get_age_group(age):
    if age <= 1:
        return 'Infant'
    elif age <= 4:
        return 'Toddler'
    elif age <= 12:
        return 'Child'
    elif age <= 19:
        return 'Teen'
    elif age <= 39:
        return 'Adult'
    elif age <= 59:
        return 'Middle_Age_Adult'
    else:
        return 'Senior_Adult'

# Gender mapping
gender_map = {0: 'Male', 1: 'Female'}

# Create subdirectories for both age and gender (train/validation + categories)
age_groups = ['Infant', 'Toddler', 'Child', 'Teen', 'Adult', 'Middle_Age_Adult', 'Senior_Adult']
gender_groups = ['Male', 'Female']

for group in age_groups:
    os.makedirs(os.path.join(age_base, 'train', group), exist_ok=True)
    os.makedirs(os.path.join(age_base, 'validation', group), exist_ok=True)

for group in gender_groups:
    os.makedirs(os.path.join(gender_base, 'train', group), exist_ok=True)
    os.makedirs(os.path.join(gender_base, 'validation', group), exist_ok=True)

# Get and shuffle image files
all_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
random.shuffle(all_files)  # Random shuffle for unbiased train/validation split

# 80-20 split
split_index = int(0.8 * len(all_files))
train_files = all_files[:split_index]
val_files = all_files[split_index:]

# Function to move files into correct folders
def process_files(file_list, subset):
    for file in file_list:
        try:
            parts = file.split('_')
            age = int(parts[0])
            gender = int(parts[1])

            age_group = get_age_group(age)
            gender_label = gender_map[gender]

            # Source path
            src_path = os.path.join(source_folder, file)

            # Destination paths
            age_dest = os.path.join(age_base, subset, age_group)
            gender_dest = os.path.join(gender_base, subset, gender_label)

            # Copy files
            shutil.copy(src_path, os.path.join(age_dest, file))
            shutil.copy(src_path, os.path.join(gender_dest, file))

        except Exception as e:
            print(f"⚠️ Error processing file {file}: {e}")

# Process both sets
process_files(train_files, 'train')
process_files(val_files, 'validation')

print("✅ Dataset successfully sorted into Age and Gender categories with balanced training and validation folders.")
