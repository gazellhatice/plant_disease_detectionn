# Kullanıcı bir yaprak görseli seçer. Model bu görsel üzerinde tahmin yapar, 
# varsa hastalık alanını kutucukla gösterir ve sonucu ekranda gösterir.
import cv2
import numpy as np
import tensorflow.lite as tflite
import tkinter as tk
from tkinter import filedialog

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # Example: [1, 128, 128, 3]

# Define labels (Update with actual disease names from your dataset)
labels = ['Apple Scab', 'Apple Cedar Rust', 'Apple Leaf Spot', 'Apple Powdery Mildew', 'Unknown','Apple Fruit Rot',
    'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Septoria Leaf Spot', 'Tomato Fusarium Wilt', 
    'Tomato Verticillium Wilt', 'Potato Late Blight', 'Potato Early Blight', 'Potato Scab', 'Corn Rust', 
    'Corn Blight', 'Corn Smut', 'Wheat Rust', 'Wheat Blight', 'Wheat Powdery Mildew', 'Pepper Bacterial Spot', 
    'Pepper Powdery Mildew', 'Strawberry Leaf Spot', 'Strawberry Powdery Mildew', 'Strawberry Botrytis Fruit Rot', 
    'Squash Blossom End Rot', 'Cabbage Worm', 'Cabbage Downy Mildew', 'Cabbage Black Rot', 'Tomato Spider Mites', 
    'Tomato Leaf Mold', 'Tomato Healthy', 'Apple Black Rot', 'Apple Fire Blight', 'Grape Black Rot', 'Grape Healthy', 
    'Peach Bacterial Spot', 'Peach Healthy', 'Soybean Rust', 'Squash Mosaic Virus', 'Rice Blast', 'Rice Sheath Blight', 
    'Rice Brown Spot', 'Rice Healthy', 'Citrus Greening', 'Citrus Healthy', 'Mango Anthracnose', 'Mango Healthy', 
    'Cotton Wilt', 'Cotton Healthy', 'Banana Black Sigatoka', 'Banana Healthy', 'Coffee Leaf Rust', 'Coffee Healthy', 
    'Pear Leaf Spot', 'Pear Fire Blight', 'Pear Healthy', 'Pomegranate Bacterial Spot', 'Pomegranate Healthy', 
    'Guava Wilt', 'Guava Healthy', 'Lettuce Downy Mildew', 'Lettuce Healthy', 'Spinach Leaf Spot', 'Spinach Healthy', 
    'Brinjal Wilt', 'Brinjal Healthy', 'Okra Yellow Vein Mosaic Virus', 'Okra Healthy', 'Zucchini Mosaic Virus', 
    'Zucchini Healthy', 'Turnip Leaf Spot', 'Turnip Healthy', 'Mustard Leaf Spot', 'Mustard Healthy', 'Kale Healthy', 
    'Tomato Blossom End Rot', 'Tomato Bacterial Wilt', 'Tomato Anthracnose', 'Tomato White Mold', 'Tomato Target Spot']  

def preprocess_image(image_path):
    """Loads and preprocesses an image for model inference."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    
    img = cv2.resize(img, (input_shape[1], input_shape[2]))  # Resize to model input size
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # Normalize
    return img

def predict_disease(image_path):
    """Runs the TFLite model on an image and returns the disease name & confidence."""
    img = preprocess_image(image_path)
    if img is None:
        return "Error", 0.0
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    class_id = np.argmax(predictions)  # Get the predicted class
    confidence = predictions[class_id]  # Get confidence score

    disease_name = labels[class_id] if class_id < len(labels) else "Unknown"
    return disease_name, confidence

def draw_bounding_box(image_path):
    """Detects affected areas and draws bounding boxes."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return image

def select_image():
    """Opens a file dialog to choose an image and runs the model on it."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        print("No image selected.")
        return

    print(f"Selected Image: {file_path}")
    
    disease, confidence = predict_disease(file_path)
    print(f"Predicted Disease: {disease} ({confidence:.2f} confidence)")

    # Draw bounding box
    processed_image = draw_bounding_box(file_path)

    # Display the result
    cv2.putText(processed_image, f"{disease} ({confidence:.2f})", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Plant Disease Detection", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    select_image()
