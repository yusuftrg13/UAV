# UAV

# UAV Güvenlik Amaçlı Nesne Tanıma Projesi

Bu proje, İnsansız Hava Araçları (UAV/Drone) için YOLOv8 mimarisi kullanılarak geliştirilmiş, hafif ve optimize edilmiş bir güvenlik odaklı nesne tespit sistemidir. Hit-UAV veri seti kullanılarak eğitilen model, insanlar, arabalar, bisikletler ve diğer araçları havadan görüntülerde yüksek doğrulukla tespit edebilmektedir.

Veri Seti:https://www.kaggle.com/datasets/pandrii000/hituav-a-highaltitude-infrared-thermal-dataset

[English version below / İngilizce versiyonu aşağıdadır](#uav-security-oriented-object-detection-project)

## 🚀 Öne Çıkan Özellikler

- **Hafif YOLOv8n Mimarisi**: Drone sistemlerinde sınırlı kaynaklarla çalışmaya uygun
- **Gerçek Zamanlı Tespit**: Düşük gecikme süresiyle güvenlik tehditleri tespiti
- **Optimizasyon Seçenekleri**: ONNX ve TensorRT desteği, FP16 yarı-hassasiyet modu
- **Güvenlik Seviye Sistemi**: Tespit edilen nesne tipine göre (HIGH/MEDIUM/LOW) güvenlik alarmı
- **Yüksek Doğruluk**: 5514 nesneyi %64.67 ortalama güven skoruyla tespit

## 📊 Test Sonuçları

### Genel Performans
- 571 test görüntüsünde 5514 nesne tespiti
- %99.12 tespit oranı (566/571 görüntü)
- %64.67 ortalama tespit güven skoru

### Sınıf Bazlı İstatistikler
| Sınıf | Tespit Sayısı | Ort. Güven Skoru |
|-------|---------------|------------------|
| İnsan | 2961 | 0.5995 (59.95%) |
| Araba | 1527 | 0.7603 (76.03%) |
| Bisiklet | 980 | 0.6110 (61.10%) |
| Diğer Araç | 25 | 0.7000 (70.00%) |
| Önemsiz | 21 | 0.6427 (64.27%) |

### YOLOv8n Model Metrikleri
- **mAP50**: 0.784
- **mAP50-95**: 0.497

## 📌 Kullanım

### Gerçek Zamanlı Tespit

```bash
python deploy_security_model.py --source 0 --save_frames
```

### Düşük Kaynaklı Sistemlerde

```bash
python deploy_security_model.py --source 0 --low_memory --imgsz 416 --half
```

### Video İşleme

```bash
python deploy_security_model.py --source video.mp4 --output sonuc.mp4 --conf 0.35
```

### Model Eğitimi

```bash
python train_security_model.py
```

### Model Değerlendirme

```bash
python evaluate_model.py
```

## 🔧 Optimizasyonlar

Drone gibi sınırlı kaynaklı sistemler için:
- **Hafif Model**: YOLOv8n versiyonu (daha küçük boyut)
- **Format Dönüşümü**: ONNX/TensorRT ile optimizasyon
- **Düşük Bellek Modu**: Kare atlama ve düşük çözünürlük desteği
- **FP16 Modu**: Daha hızlı çıkarım için yarı-hassasiyet
- **Dinamik Boyutlandırma**: Sistem kapasitesine uygun boyutlandırma

## 🔍 Güvenlik Uygulamaları

- Yetkisiz alan erişim tespiti
- Sınır güvenliği izleme
- Araç takibi ve sayımı
- Kapalı alan güvenlik taraması
- Kalabalık analizi ve davranış tespiti

## 📦 Gereksinimler

```bash
pip install ultralytics opencv-python numpy matplotlib torch tqdm
```

## 📸 Örnek Görseller

![0_120_60_0_09564](https://github.com/user-attachments/assets/b87aa3df-44be-4314-8283-7310b0d320af)
![1_70_70_0_07730](https://github.com/user-attachments/assets/8a528079-5798-48a4-ba3c-2a928f6e7449)
![alert_20250701_231416_1](https://github.com/user-attachments/assets/80a81b69-a7e0-45dc-8b2f-e940d47e7ffe)
![val_batch0_pred](https://github.com/user-attachments/assets/df27e386-327f-4c4a-b251-7f9cb87e67ea)
![confusion_matrix_normalized](https://github.com/user-attachments/assets/198deac0-2744-4bf2-abb9-8098d03d3543)


---

# UAV Security-Oriented Object Detection Project

This project presents a lightweight and optimized security-focused object detection system for Unmanned Aerial Vehicles (UAVs/Drones) using the YOLOv8 architecture. Trained on the Hit-UAV dataset, the model can accurately detect humans, cars, bicycles, and other vehicles in aerial imagery.

Veri Seti:https://www.kaggle.com/datasets/pandrii000/hituav-a-highaltitude-infrared-thermal-dataset

## 🚀 Key Features

- **Lightweight YOLOv8n Architecture**: Suitable for resource-constrained drone systems
- **Real-time Detection**: Security threat identification with low latency
- **Optimization Options**: ONNX and TensorRT support, FP16 half-precision mode
- **Security Level System**: Security alerts (HIGH/MEDIUM/LOW) based on detected object type
- **High Accuracy**: 5514 objects detected with 64.67% average confidence score

## 📊 Test Results

### General Performance
- 5514 object detections across 571 test images
- 99.12% detection rate (566/571 images)
- 64.67% average detection confidence score

### Class-Based Statistics
| Class | Detection Count | Avg. Confidence Score |
|-------|-----------------|----------------------|
| Person | 2961 | 0.5995 (59.95%) |
| Car | 1527 | 0.7603 (76.03%) |
| Bicycle | 980 | 0.6110 (61.10%) |
| Other Vehicle | 25 | 0.7000 (70.00%) |
| DontCare | 21 | 0.6427 (64.27%) |

### YOLOv8n Model Metrics
- **mAP50**: 0.784
- **mAP50-95**: 0.497

## 📌 Usage

### Real-time Detection

```bash
python deploy_security_model.py --source 0 --save_frames
```

### For Low-Resource Systems

```bash
python deploy_security_model.py --source 0 --low_memory --imgsz 416 --half
```

### Video Processing

```bash
python deploy_security_model.py --source video.mp4 --output result.mp4 --conf 0.35
```

### Model Training

```bash
python train_security_model.py
```

### Model Evaluation

```bash
python evaluate_model.py
```

## 🔧 Optimizations

For resource-constrained systems like drones:
- **Lightweight Model**: YOLOv8n version (smaller footprint)
- **Format Conversion**: ONNX/TensorRT optimization
- **Low Memory Mode**: Frame skipping and lower resolution support
- **FP16 Mode**: Half-precision for faster inference
- **Dynamic Sizing**: Resizing based on system capacity

## 🔍 Security Applications

- Unauthorized area access detection
- Border security monitoring
- Vehicle tracking and counting
- Closed area security scanning
- Crowd analysis and behavior detection

## 📦 Requirements

```bash
pip install ultralytics opencv-python numpy matplotlib torch tqdm
```

## 📸 Sample Images
![0_120_60_0_09564](https://github.com/user-attachments/assets/0213aa5d-1928-4e60-ace3-2627ed1a3ff3)
![1_70_70_0_07730](https://github.com/user-attachments/assets/448a357f-0d76-415c-b3ec-15f7f7e56e45)
![alert_20250701_231416_1](https://github.com/user-attachments/assets/e91f8242-2767-406b-b4d4-6e4818ed470d)
![val_batch0_pred](https://github.com/user-attachments/assets/d674e583-85ed-4fab-9710-9c7027fd8aff)
![confusion_matrix_normalized](https://github.com/user-attachments/assets/d67498c6-0bd2-4fec-a66b-6d4e3e7252ac)

