#belirtilen klasörün varlığını kontrol eder; 
# klasör varsa içindeki .jpg, .png veya .jpeg uzantılı görselleri listeler 
# ve ilk 5 tanesini ekrana yazdırır

import os

# ✅ Update this path based on your actual folder name
IMAGE_DIR = r"C:\Users\Hatice\OneDrive\Documents\plant detection new\Plant_leave_diseases_dataset_with_augmentation"

# ✅ Check if the folder exists
if not os.path.exists(IMAGE_DIR):
    raise ValueError(f"❌ Dataset directory does not exist: {IMAGE_DIR}")

# ✅ List available image files
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(("jpg", "png", "jpeg"))]

if not image_files:
    raise ValueError("❌ No image files found! Ensure images are inside the correct folder.")

print("✅ Found images:", image_files[:5])  # Show first 5 images
