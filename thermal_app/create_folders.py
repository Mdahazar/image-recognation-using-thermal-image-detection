import os

# Create main dataset folder
dataset_path = "thermal_dataset"
os.makedirs(dataset_path, exist_ok=True)

# List of 15 categories
categories = [
    "horse", "human", "bottle", "car", "dog",
    "cat", "tree", "building", "electronics", "machinery",
    "animal", "plant", "vehicle", "equipment", "unknown"
]

# Create all category folders
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created: {folder_path}")

print("All 15 folders created successfully!")