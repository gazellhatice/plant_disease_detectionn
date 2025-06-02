# 🌿 Plant Disease Detection Web Application

Bu proje, bitki yapraklarındaki hastalıkları derin öğrenme modelleri yardımıyla tespit eden bir web uygulamasıdır. Kullanıcılar web arayüzü üzerinden bir yaprak görseli yükleyerek yaprağın sağlıklı mı yoksa hasta mı olduğunu öğrenebilir. Sonuçlar sadece ekranda gösterilmez, aynı zamanda PDF formatında raporlanarak kullanıcıya sunulur.

---

## 📌 Özellikler

- 📷 Görsel yükleme arayüzü
- 🧠 PyTorch ve TensorFlow Lite formatında eğitilmiş modeller ile tahmin
- 📄 Tahmin sonucunun PDF rapor olarak indirilmesi
- 🔥 Kullanıcı dostu Flask tabanlı web uygulaması
- 🗂️ Çok sayıda veri seti ve etiketlenmiş örnekler ile desteklenmiş model
- 💾 Basit kurulum, yerel çalıştırma desteği

---

## 🛠️ Kullanılan Teknolojiler

| Katman        | Teknolojiler                             |
|---------------|------------------------------------------|
| Backend       | Python, Flask                            |
| Frontend      | HTML, CSS                                |
| ML Model      | PyTorch (.pt), TensorFlow Lite (.tflite) |
| Diğer         | OpenCV, Pandas, PDFKit                   |

---

## 📁 Klasör Yapısı

plant-new-2/
├── app.py                         # Ana Flask uygulaması
├── app_2.py                       # Alternatif model kontrol dosyası
├── best.pt                        # PyTorch eğitilmiş model
├── best.tflite                    # TensorFlow Lite modeli
├── new_plant_disease_model.tflite# Alternatif TFLite modeli
├── dataset/                       # Eğitim verileri klasörü
├── dataset2/                      # İkinci veri kümesi
├── dataset3/                      # Üçüncü veri kümesi
├── logs/                          # Eğitim süreci logları
├── static/                        # Statik dosyalar (PDF, CSS, img)
├── styles/                        # CSS dosyaları
├── templates/                     # HTML şablonları (Flask için)
├── uploads/                       # Kullanıcı yüklemeleri
├── labeled_dataset.csv            # Etiketli veri bilgileri
├── labeled_images.csv             # Görsel yolları ve etiketler
├── .gitignore                     # Git dışlama kuralları
└── README.md                      # Bu dosya

---

## 🚀 Kurulum Talimatları

### 1. Projeyi klonla:

```bash
git clone https://github.com/gazellhatice/plant_disease_detectionn.git
cd plant_disease_detectionn
```

### 2. Sanal ortam oluştur (opsiyonel ama önerilir):

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Gerekli paketleri yükle:

```bash
pip install -r requirements.txt
```

Eğer `requirements.txt` dosyası yoksa:

```bash
pip install flask torch torchvision tensorflow opencv-python pandas
```

### 4. Uygulamayı başlat:

```bash
python app.py
```

Tarayıcıda aç:  
👉 `http://127.0.0.1:5000/`

---

## 🔍 Nasıl Çalışır?

- Kullanıcı bir yaprak fotoğrafı yükler.
- Görsel `uploads/` klasörüne kaydedilir.
- Model, resmi analiz eder ve sınıf tahmini yapar.
- Tahmin sonucu web arayüzünde gösterilir.
- Ayrıca PDF raporu otomatik oluşturulur ve indirme bağlantısı sunulur.

---

## 📊 Model Hakkında

- Kullanılan modeller CNN tabanlıdır.
- `best.pt` PyTorch ile eğitilmiştir.
- `best.tflite` mobil/edge cihazlar için optimize edilmiştir.
- Eğitim verileri `dataset/`, `dataset2/`, `dataset3/` klasörlerinde bulunur.
- Eğitim logları `logs/` klasöründe kayıtlıdır.

---

## 📄 PDF Raporlama

- Her tahmin işlemi sonrası PDF raporu oluşturulur.
- `static/pdf_reports/` klasöründe saklanır.
- Raporlar yaprağın sınıfı, yükleme zamanı ve görsel ile birlikte detaylandırılır.

---

## 🎯 Geliştirme Planları

- [ ] Daha fazla bitki ve hastalık sınıfı ekleme
- [ ] Mobil cihazlar için özel arayüz
- [ ] Canlı kamera desteği ile gerçek zamanlı tahmin
- [ ] Model eğitimi için kullanıcıdan geri bildirim toplama

---

## 🤝 Katkı Sağlamak

Katkı yapmak için şu adımları izleyebilirsin:

1. Bu repoyu forkla 🍴
2. Yeni bir dal oluştur: `git checkout -b yeni-özellik`
3. Değişikliklerini commit et: `git commit -m "Yeni özellik eklendi"`
4. Branch’i gönder: `git push origin yeni-özellik`
5. Pull Request (PR) oluştur

---

## 📜 Lisans

Bu proje MIT lisansı ile lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasına göz atabilirsiniz.