import cv2
import numpy as np
import tensorflow.lite as tflite

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the expected input shape
input_shape = input_details[0]['shape'] 

# Define labels
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

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply Gaussian blur to reduce noise
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Convert to grayscale and resize to model input size
    img = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[2]))  

    # Normalize input
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    # Get prediction results
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predictions = np.exp(predictions) / np.sum(np.exp(predictions))  # Softmax Scaling

    # Get highest confidence class
    class_id = np.argmax(predictions)
    confidence = predictions[class_id]

    # Validate class_id
    label = labels[class_id] if class_id < len(labels) else "Unknown"
    
    # Get frame dimensions
    h, w, _ = frame.shape
    startX, startY, endX, endY = int(w * 0.15), int(h * 0.15), int(w * 0.75), int(h * 0.75)

    # Define color based on confidence scoreq
    color = (0, 255, 0) if confidence > 0.75 else (0, 0, 255)

    # Draw bounding box
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
    cv2.putText(frame, f"Disease : {label}  (Acc: {confidence:.2f})", 
                (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    # Display output
    cv2.imshow("Plant Disease Detection", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
