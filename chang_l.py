import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.cluster import KMeans
import shutil
import matplotlib.pyplot as plt

# ✅ Fix path issue
IMAGE_DIR = r"Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network\Plant_leave_diseases_dataset_with_augmentation"

# Load MobileNetV2 model
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# ✅ Get all image paths
image_paths = [os.path.join(IMAGE_DIR, img) for img in os.listdir(IMAGE_DIR) if img.endswith(("jpg", "png", "jpeg"))]

if not image_paths:
    raise ValueError("⚠️ Veri kümesi dizininde resim bulunamadı!")

# ✅ Extract features
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return base_model.predict(img_array)[0]
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return None

# ✅ Filter out failed extractions
features = np.array([f for f in (extract_features(img) for img in image_paths) if f is not None])

if features.shape[0] == 0:
    raise ValueError("❌ Geçerli özellikler ayıklanmadı! Veri kümesini ve görüntü formatlarını kontrol edin.")

# ✅ Clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(features)

# ✅ Organize images into folders
output_dir = "labeled_dataset"
os.makedirs(output_dir, exist_ok=True)

for i, img_path in enumerate(image_paths):
    cluster_label = labels[i]
    cluster_folder = os.path.join(output_dir, f"cluster_{cluster_label}")
    os.makedirs(cluster_folder, exist_ok=True)
    shutil.copy(img_path, cluster_folder)

print("✅ Otomatik etiketleme tamamlandı! 'Labeled_dataset' klasörünü kontrol edin.")