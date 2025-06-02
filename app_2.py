from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
import tensorflow.lite as tflite
import google.generativeai as genai
from reportlab.pdfgen import canvas
import cv2
import numpy as np
import tensorflow.lite as tflite
import tkinter as tk
from tkinter import filedialog
from urllib.parse import unquote
from flask import Response

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Google Gemini API
genai.configure(api_key="AIzaSyC-s6YUbTogOI_PXhjMF5HIi99erIV_TM0")

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
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

# Define disease prevention advice


advice_dict = {
    'Apple Scab': 'Yaprakları etkileyen mantar hastalığına karşı uygun ilaçlar kullanın ve enfekte yaprakları temizleyin.',
    'Apple Cedar Rust': 'Elma ağaçlarını ardıç ağaçlarından uzak tutun ve uygun fungisit uygulayın.',
    'Apple Leaf Spot': 'Hastalıklı yaprakları uzaklaştırın ve çevresel nemi azaltacak şekilde sulama yapın.',
    'Apple Powdery Mildew': 'İyi hava sirkülasyonu sağlayın ve sülfür bazlı fungisitler kullanın.',
    'Unknown': 'Tanımlanamayan hastalık. Daha fazla analiz yapılmalı veya uzmana başvurulmalı.',
    'Apple Fruit Rot': 'Enfekte meyveleri toplayın ve ağaç altı hijyenine dikkat edin.',
    'Tomato Early Blight': 'Eski yaprakları temizleyin ve uygun fungusit kullanın.',
    'Tomato Late Blight': 'Nemli koşullardan kaçının, dirençli çeşitler tercih edin.',
    'Tomato Septoria Leaf Spot': 'Alt yaprakları budayın ve hastalıklı alanları yok edin.',
    'Tomato Fusarium Wilt': 'Toprak solarizasyonu uygulayın ve dayanıklı çeşitler kullanın.',
    'Tomato Verticillium Wilt': 'Toprağı dezenfekte edin, dönüşümlü ekim yapın.',
    'Potato Late Blight': 'Yaprak ıslaklığını önleyin, erken sabah sulama yapın.',
    'Potato Early Blight': 'Hasta bitki kalıntılarını temizleyin, azot dengesine dikkat edin.',
    'Potato Scab': 'Toprak pH’ını düşürün, çok fazla gübre kullanımından kaçının.',
    'Corn Rust': 'Hastalıklı yaprakları yok edin, dirençli türleri tercih edin.',
    'Corn Blight': 'Tarla hijyenine dikkat edin, doğru ekim aralığı bırakın.',
    'Corn Smut': 'Enfekte koçanları erken dönemde çıkarın, dayanıklı çeşit kullanın.',
    'Wheat Rust': 'İlkbahar aylarında dikkatli olun, fungisit uygulaması yapın.',
    'Wheat Blight': 'Hastalık belirtileri için erken tarama yapın.',
    'Wheat Powdery Mildew': 'Yoğun ekimden kaçının, mantar ilaçları kullanın.',
    'Pepper Bacterial Spot': 'Hasta bitkileri imha edin ve sulamayı dipten yapın.',
    'Pepper Powdery Mildew': 'Fungisit kullanın ve nemi azaltın.',
    'Strawberry Leaf Spot': 'Hasta yaprakları kesin, yaprak altı ilaç uygulayın.',
    'Strawberry Powdery Mildew': 'Hava akışı sağlayın ve uygun ilaç kullanın.',
    'Strawberry Botrytis Fruit Rot': 'Meyve teması olan yaprakları kaldırın, erken hasat yapın.',
    'Squash Blossom End Rot': 'Kalsiyum eksikliğini giderin ve düzensiz sulamadan kaçının.',
    'Cabbage Worm': 'Elle toplayın veya biyolojik mücadele yöntemleri kullanın.',
    'Cabbage Downy Mildew': 'Geç hasat yapmayın ve sabah sulamayı tercih edin.',
    'Cabbage Black Rot': 'Tohumları dezenfekte edin ve hastalıklı bitkileri uzaklaştırın.',
    'Tomato Spider Mites': 'Yaprak altlarını kontrol edin ve doğal yırtıcılar kullanın.',
    'Tomato Leaf Mold': 'Serada nemi düşürün ve havalandırmayı artırın.',
    'Tomato Healthy': 'Bitki sağlıklı. Düzenli bakım ve sulamaya devam edin.',
    'Apple Black Rot': 'Enfekte meyve ve dalları kesip uzaklaştırın.',
    'Apple Fire Blight': 'İlkbaharda hızlıca müdahale edin, enfekte dalları kesin.',
    'Grape Black Rot': 'Bağ temizliği yapın ve önleyici fungisit uygulayın.',
    'Grape Healthy': 'Üzüm sağlıklı. Bakımı sürdürün ve zararlılara karşı dikkatli olun.',
    'Peach Bacterial Spot': 'Bakır bazlı ilaçlar kullanın ve yaprak teması azaltın.',
    'Peach Healthy': 'Şeftali sağlıklı. Mevsimsel gübrelemeyi unutmayın.',
    'Soybean Rust': 'Erken tespit önemlidir, uygun ilaçlarla mücadele edin.',
    'Squash Mosaic Virus': 'Vektör böceklerle mücadele edin, enfekte bitkileri çıkarın.',
    'Rice Blast': 'Yüksek azot kullanımından kaçının, dirençli çeşitleri tercih edin.',
    'Rice Sheath Blight': 'Yüksek nemli bölgelerde dikkatli olun.',
    'Rice Brown Spot': 'Dengeli gübreleme yapın ve erken dönemde ilaçlama yapın.',
    'Rice Healthy': 'Pirinç sağlıklı. Su kontrolüne ve haşereye dikkat edin.',
    'Citrus Greening': 'Vektör böceklerle mücadele edin, hastalıklı ağaçları sökün.',
    'Citrus Healthy': 'Ağaç sağlıklı. Düzenli budama ve ilaçlamaya devam edin.',
    'Mango Anthracnose': 'Yağış sonrası mantar ilaçları uygulayın.',
    'Mango Healthy': 'Mango sağlıklı. Çiçeklenme dönemine dikkat edin.',
    'Cotton Wilt': 'Toprakta nem dengesine dikkat edin, dayanıklı çeşit kullanın.',
    'Cotton Healthy': 'Pamuk sağlıklı. Yabancı otları temiz tutun.',
    'Banana Black Sigatoka': 'Yaprakları erken dönemde budayın, mantar ilacı uygulayın.',
    'Banana Healthy': 'Muz bitkisi sağlıklı. Rutin kontrolleri ihmal etmeyin.',
    'Coffee Leaf Rust': 'Yaprakları düzenli kontrol edin, gölge yoğunluğunu azaltın.',
    'Coffee Healthy': 'Kahve bitkisi sağlıklı. Gölge ve su dengesine dikkat edin.',
    'Pear Leaf Spot': 'Hasta yaprakları toplayın ve uygun ilaç kullanın.',
    'Pear Fire Blight': 'Bulaşık dalları erken dönemde budayın.',
    'Pear Healthy': 'Armut sağlıklı. Erken çiçeklenmede dondan koruyun.',
    'Pomegranate Bacterial Spot': 'Enfekte meyveleri toplayın, uygun bakır içerikli ilaç kullanın.',
    'Pomegranate Healthy': 'Nar sağlıklı. Ağaç arası mesafeye dikkat edin.',
    'Guava Wilt': 'Drenajı iyi olan toprak tercih edin, sulama dikkatli yapılmalı.',
    'Guava Healthy': 'Guava sağlıklı. Fazla sulamaktan kaçının.',
    'Lettuce Downy Mildew': 'Serin ve nemli havalarda ilaçlama yapın.',
    'Lettuce Healthy': 'Marul sağlıklı. Zararlılara karşı düzenli kontrol yapılmalı.',
    'Spinach Leaf Spot': 'Hastalıklı yaprakları koparın, serin ortam sağlayın.',
    'Spinach Healthy': 'Ispanak sağlıklı. Toprak pH dengesine dikkat edin.',
    'Brinjal Wilt': 'Toprağı dezenfekte edin ve enfekte bitkileri imha edin.',
    'Brinjal Healthy': 'Patlıcan sağlıklı. Destek çubukları kullanmayı unutmayın.',
    'Okra Yellow Vein Mosaic Virus': 'Vektör haşerelerden korunmak için insektisit kullanın.',
    'Okra Healthy': 'Bamya sağlıklı. Çiçeklenme döneminde sulamaya dikkat.',
    'Zucchini Mosaic Virus': 'Yaprak bitleriyle mücadele edin ve enfekte bitkileri ayırın.',
    'Zucchini Healthy': 'Kabak sağlıklı. Erken hasat verimi artırır.',
    'Turnip Leaf Spot': 'Enfekte yaprakları temizleyin ve mantar ilaçları kullanın.',
    'Turnip Healthy': 'Şalgam sağlıklı. Düzenli sulama ve çapalama yapılmalı.',
    'Mustard Leaf Spot': 'Yaprakları kontrol edin ve ilkbaharda ilaç uygulayın.',
    'Mustard Healthy': 'Hardal sağlıklı. Çiçeklenme dönemine dikkat.',
    'Kale Healthy': 'Kıvırcık lahana sağlıklı. Güneş alan bölgede yetiştirilmeli.',
    'Tomato Blossom End Rot': 'Kalsiyum takviyesi yapın, sulamayı düzenli hale getirin.',
    'Tomato Bacterial Wilt': 'Hasta bitkileri sökün, toprağı dezenfekte edin.',
    'Tomato Anthracnose': 'Hasattan önce mantar ilacı uygulayın.',
    'Tomato White Mold': 'Hastalık görülen yaprakları budayın, serin havalarda dikkatli olun.',
    'Tomato Target Spot': 'Alt yaprakları budayın ve fungusit uygulayın.'
}


def get_gemini_advice(disease):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Provide detailed prevention and treatment advice for {disease} in plants."

    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and hasattr(response, 'text') else "No advice available."
    except Exception as e:
        return f"Error fetching advice: {str(e)}"

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (input_shape[1], input_shape[2]))
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img

def predict_disease(image_path, advice_dict=None):
    img = preprocess_image(image_path)
    if img is None:
        return "Error", 0.0, "No image found."

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    class_id = np.argmax(predictions)
    confidence = predictions[class_id]
    disease_name = labels[class_id] if class_id < len(labels) else "Unknown"
    prevention_advice = advice_dict.get(disease_name, get_gemini_advice(disease_name))
    return disease_name, confidence, prevention_advice

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')  # about.html adında bir dosya olması gerekir

@app.route('/contact')
def contact():
    return render_template('contact.html')  
    


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and preprocess image
            image = cv2.imread(filepath)
            image = cv2.resize(image, (input_shape[1], input_shape[2]))
            image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Get prediction
            prediction_index = np.argmax(output_data)
            prediction = labels[prediction_index]
            confidence = float(output_data[0][prediction_index])
            advice = advice_dict.get(prediction, "No specific advice available.")

            return render_template('upload.html', filename=filename, prediction=prediction, confidence=confidence, advice=advice)
    
    return render_template('upload.html')


@app.route('/download_report', methods=['POST'])
def download_report():
    filename = request.form['filename']
    prediction = request.form['prediction']
    confidence = request.form['confidence']
    advice = request.form['advice']
    language = request.form.get('language', 'en')  # varsa al, yoksa "en" 
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "report.pdf")
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 16)
    
    # Title
    c.drawString(200, 800, "🌱 Plant Disease Detection Report")
    c.line(50, 790, 550, 790)  # Horizontal line
    print(end="\n\n")
    
    # Add Image
    try:
        print(end="\n")
        c.drawImage(image_path, 180, 500, width=250, height=170)  # Adjusted image position and size
    except Exception as e:
        c.drawString(100, 610, "Image could not be loaded.")

    # Disease Prediction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 460, f"🦠 Disease Detected: {prediction}")

    # Confidence Score
    c.setFont("Helvetica", 12)
    c.drawString(100, 440, f"Confidence Score: {float(confidence):.2f}")

    # Prevention Advice Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 420, "----> Prevention & Treatment Advice:")

    # Prevention Advice Content
    c.setFont("Helvetica", 12)
    y = 400
    for line in advice.split('. '):
        c.drawString(120, y, f"- {line.strip()}.")
        y -= 20

    c.save()
    
    return send_file(pdf_path, as_attachment=True)

@app.route('/live')
def live_page():
    return render_template('live.html')
@app.route('/live', methods=['GET'])
def live():
    return render_template('live.html')  # Render a new template for live detection.

def start_live_detection():
        # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# Get the expected input shape
    input_shape = input_details[0]['shape'] 


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

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess and predict
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        class_id = np.argmax(predictions)
        confidence = predictions[class_id]
        label = labels[class_id] if class_id < len(labels) else "Unknown"

        # Draw results on frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    
if __name__ == '__main__':
    app.run(debug=True)
