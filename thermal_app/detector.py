import cv2
import numpy as np
import joblib
import os

class ThermalObjectDetector:
    def __init__(self, model_path='thermal_model.pkl', classes_path='classes.npy'):
        self.model = None
        self.classes = []
        self.model_loaded = False
        self.load_model(model_path, classes_path)
    
    def load_model(self, model_path='thermal_model.pkl', classes_path='classes.npy'):
        """Load the trained model and classes"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return False
                
            if not os.path.exists(classes_path):
                print(f"❌ Classes file not found: {classes_path}")
                return False
            
            self.model = joblib.load(model_path)
            self.classes = np.load(classes_path, allow_pickle=True)
            self.model_loaded = True
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"📋 Available classes: {list(self.classes)}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def extract_features(self, img):
        """Extract features from thermal image (must match trainer)"""
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            # Resize to match training size
            img_resized = cv2.resize(img, (128, 128))
            gray_resized = cv2.resize(gray, (128, 128))
            
            # Resize to smaller size for feature extraction
            img_small = cv2.resize(img_resized, (64, 64))
            gray_small = cv2.resize(gray_resized, (64, 64))
            
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
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None
    
    def predict_object(self, image_path):
        """Predict object class from thermal image"""
        if not self.model_loaded or self.model is None:
            return "Model not loaded", 0.0
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                return "Could not load image", 0.0
            
            # Extract features
            features = self.extract_features(img)
            if features is None:
                return "Feature extraction failed", 0.0
                
            features = features.reshape(1, -1)
            
            # Predict - handle both model types
            if hasattr(self.model, 'category_name'):  # SingleCategoryModel
                prediction = self.model.predict(features)[0]
                probabilities = self.model.predict_proba(features)[0]
                confidence = probabilities[0]  # Single category confidence
            else:
                # Standard SVM model
                prediction = self.model.predict(features)[0]
                probabilities = self.model.predict_proba(features)[0]
                confidence = np.max(probabilities)
            
            return prediction, confidence
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Prediction failed", 0.0

def test_detector():
    """Test the detector with a sample image"""
    detector = ThermalObjectDetector()
    
    if detector.model_loaded:
        test_image = "test_thermal.jpg"
        if os.path.exists(test_image):
            prediction, confidence = detector.predict_object(test_image)
            print(f"🎯 Prediction: {prediction}")
            print(f"📊 Confidence: {confidence:.2%}")
        else:
            print("ℹ️  Test image not found!")
    else:
        print("❌ Detector not loaded properly")

if __name__ == "__main__":
    test_detector()