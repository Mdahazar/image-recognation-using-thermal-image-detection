import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from collections import Counter

class ThermalObjectTrainer:
    def __init__(self):
        self.model = None
        self.classes = []
        self.features = []
        self.labels = []
        
    def load_images_from_folders(self, base_path):
        """
        Load thermal images from category folders
        Works with ANY number of categories (1, 2, 3, ... 15)
        """
        self.features = []
        self.labels = []
        self.classes = []
        
        # Get all category folders
        categories = [d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d))]
        categories.sort()
        
        print(f"Found categories: {categories}")
        
        # Count images in each category
        category_counts = {}
        for category in categories:
            category_path = os.path.join(base_path, category)
            image_count = 0
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_count += 1
            category_counts[category] = image_count
            print(f"{category}: {image_count} images")
        
        # Include ALL categories with at least 1 image
        valid_categories = [cat for cat in categories if category_counts[cat] >= 1]
        
        if len(valid_categories) == 0:
            print("ERROR: No folders contain images!")
            return False
        
        self.classes = valid_categories
        print(f"\nTraining with {len(valid_categories)} valid categories: {valid_categories}")
        
        # Load images from valid categories
        for category in valid_categories:
            category_path = os.path.join(base_path, category)
            print(f"Loading {category}...")
            
            loaded_count = 0
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(category_path, filename)
                    
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize image to standard size
                        img = cv2.resize(img, (128, 128))
                        
                        # Extract features
                        features = self.extract_features_simple(img)
                        
                        self.features.append(features)
                        self.labels.append(category)
                        loaded_count += 1
            
            print(f"  Loaded {loaded_count} images from {category}")
        
        print(f"\nTotal loaded: {len(self.features)} images across {len(self.classes)} categories")
        return True
        
    def extract_features_simple(self, img):
        """Simplified feature extraction"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Resize to smaller size for faster processing
        img_small = cv2.resize(img, (64, 64))
        gray_small = cv2.resize(gray, (64, 64))
        
        # 1. Simple color histogram
        hist = cv2.calcHist([img_small], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        hist_features = cv2.normalize(hist, hist).flatten()
        
        # 2. Simple HOG features
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(gray_small)
        hog_features = hog_features.flatten()
        
        # 3. Basic texture features
        mean_val = np.mean(gray_small)
        std_val = np.std(gray_small)
        
        # Combine all features
        combined_features = np.concatenate([hist_features, hog_features, [mean_val, std_val]])
        
        return combined_features
    
    def train_model(self):
        """Train model that works with ANY number of categories"""
        if len(self.features) == 0:
            print("No features to train on!")
            return False
            
        # Convert to numpy arrays
        X = np.array(self.features)
        y = np.array(self.labels)
        
        print(f"Training data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Check label distribution
        label_counts = Counter(y)
        print("Label distribution:", label_counts)
        
        # Handle ALL cases: 1 category, 2 categories, 3 categories, etc.
        if len(label_counts) == 1:
            # Single category - create a special model
            print("Training single-category model...")
            self.model = SingleCategoryModel(list(label_counts.keys())[0], X)
            print(f"✅ Single category model trained for: {list(label_counts.keys())[0]}")
        else:
            # Multiple categories - use standard SVM
            print("Training multi-category model...")
            
            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError as e:
                print(f"Stratified split failed: {e}")
                print("Using random split instead...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Train SVM classifier
            self.model = SVC(kernel='linear', probability=True, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Multi-category model trained successfully!")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Classes learned: {list(self.model.classes_)}")
        
        return True
    
    def save_model(self, model_path='thermal_model.pkl', classes_path='classes.npy'):
        """Save trained model and classes"""
        if self.model is not None:
            joblib.dump(self.model, model_path)
            np.save(classes_path, self.classes)
            print(f"✅ Model saved to {model_path}")
            print(f"✅ Classes saved to {classes_path}")
            print(f"📋 Model can detect: {self.classes}")
            return True
        else:
            print("❌ No model to save!")
            return False

class SingleCategoryModel:
    """Simple model for single category detection"""
    def __init__(self, category_name, training_features):
        self.category_name = category_name
        self.classes_ = [category_name]
        # Store feature statistics for similarity comparison
        self.avg_features = np.mean(training_features, axis=0)
        self.std_features = np.std(training_features, axis=0)
        
    def predict(self, X):
        """Always predict the single category"""
        return np.array([self.category_name] * len(X))
    
    def predict_proba(self, X):
        """Return confidence based on feature similarity"""
        probas = []
        for features in X:
            # Calculate similarity to average training features
            if np.all(self.std_features == 0):
                # If all features are the same, return high confidence
                confidence = 0.9
            else:
                # Calculate normalized distance
                distance = np.sqrt(np.sum(((features - self.avg_features) / (self.std_features + 1e-8)) ** 2))
                # Convert distance to confidence (closer = higher confidence)
                confidence = max(0.3, min(0.95, 1.0 - (distance / 100.0)))
            probas.append([confidence])
        return np.array(probas)

def main():
    # Initialize trainer
    trainer = ThermalObjectTrainer()
    
    # Path to your dataset folder
    dataset_path = "thermal_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder '{dataset_path}' not found!")
        print("Please create the thermal_dataset folder with category subfolders.")
        return
    
    # Load images and train model
    success = trainer.load_images_from_folders(dataset_path)
    
    if success and len(trainer.features) > 0:
        if trainer.train_model():
            if trainer.save_model():
                print("\n🎯 TRAINING COMPLETED SUCCESSFULLY!")
                if len(trainer.classes) == 1:
                    print(f"📊 Model trained to detect: {trainer.classes[0]}")
                else:
                    print(f"📊 Model trained to detect {len(trainer.classes)} categories: {trainer.classes}")
            else:
                print("❌ Failed to save model!")
        else:
            print("❌ Training failed!")
    else:
        print("❌ No valid images found to train!")

if __name__ == "__main__":
    main()