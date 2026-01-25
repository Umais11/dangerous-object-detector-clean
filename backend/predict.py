import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Configuration
MODEL_PATH = '../model/saved_models/best_model.h5'
IMG_SIZE = 128
CLASS_NAMES = ['gun', 'knife']


class DangerousObjectDetector:
    """
    Class for dangerous object detection
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.class_names = CLASS_NAMES
        self.img_size = IMG_SIZE
        self.load_model()

    def load_model(self):
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = load_model(self.model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    # 🔥🔥🔥 THIS WAS MISSING INSIDE CLASS
    def predict(self, img_path):
        """
        Predict Dangerous vs Safe
        """
        if self.model is None:
            return {'success': False, 'error': 'Model not loaded'}

        try:
            processed_img = self.preprocess_image(img_path)

            prediction = float(self.model.predict(processed_img, verbose=0)[0][0])

            # sigmoid logic
            # 0 → dangerous | 1 → safe
            if prediction < 0.85:
                predicted_class = "dangerous"
                confidence = (1 - prediction) * 100
                is_dangerous = True
            else:
                predicted_class = "safe"
                confidence = prediction * 100
                is_dangerous = False

            probabilities = {
                "dangerous": round((1 - prediction) * 100, 2),
                "safe": round(prediction * 100, 2)
            }

            return {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'is_dangerous': is_dangerous,
                'probabilities': probabilities
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}


def test_prediction(image_path):
    print("=" * 50)
    print("DANGEROUS OBJECT DETECTION - TEST")
    print("=" * 50)

    detector = DangerousObjectDetector()

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    print(f"\n📷 Testing image: {image_path}")

    result = detector.predict(image_path)

    if result['success']:
        print("\n✅ PREDICTION RESULTS:")
        print(f"   Class: {result['predicted_class'].upper()}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Dangerous: {'YES' if result['is_dangerous'] else 'NO'}")

        print("\n📊 All Class Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"   {cls}: {prob:.2f}%")
    else:
        print(f"\n❌ Prediction failed: {result['error']}")

    print("=" * 50)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_prediction(sys.argv[1])
    else:
        print("Usage: python predict.py <image_path>")
