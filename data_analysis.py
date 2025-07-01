import os
import random
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO

# Veri seti klasör yolları
DATA_ROOT = r'c:\Users\Casper\Desktop\UAV\hit-uav'
IMAGES_TRAIN = os.path.join(DATA_ROOT, 'images', 'train')
LABELS_TRAIN = os.path.join(DATA_ROOT, 'labels', 'train')
IMAGES_VAL = os.path.join(DATA_ROOT, 'images', 'val')
LABELS_VAL = os.path.join(DATA_ROOT, 'labels', 'val')
IMAGES_TEST = os.path.join(DATA_ROOT, 'images', 'test')
LABELS_TEST = os.path.join(DATA_ROOT, 'labels', 'test')

# Sınıf adları
class_names = {
    0: 'Person',
    1: 'Car',
    2: 'Bicycle',
    3: 'OtherVehicle',
    4: 'DontCare'
}

def load_labels(label_path):
    """Etiket dosyasını okur ve içeriğini döndürür"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([class_id, x_center, y_center, width, height])
    return np.array(labels) if labels else np.array([])

def analyze_dataset(images_dir, labels_dir):
    """Veri seti istatistiklerini analiz eder"""
    class_counts = Counter()
    image_sizes = []
    bbox_aspect_ratios = []
    bbox_areas = []
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    
    print(f"Toplam görüntü sayısı: {total_images}")
    
    # Rastgele 100 görüntü seçerek istatistik toplayalım
    sample_size = min(100, total_images)
    sampled_images = random.sample(image_files, sample_size)
    
    for img_file in sampled_images:
        # Görüntü dosyası ve karşılık gelen etiket dosyasının yollarını alalım
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
        
        # Görüntüyü oku ve boyutunu kaydet
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            image_sizes.append((w, h))
            
            # Etiketleri oku
            labels = load_labels(label_path)
            if len(labels) > 0:
                # Sınıfların sayısını güncelle
                for label in labels:
                    class_id = int(label[0])
                    class_counts[class_id] += 1
                    
                    # Bounding box özelliklerini hesapla
                    # Normalize edilmiş değerleri gerçek piksel değerlerine çevir
                    bbox_w = label[3] * w
                    bbox_h = label[4] * h
                    
                    # Aspect ratio ve alan hesapla
                    if bbox_h > 0:
                        aspect_ratio = bbox_w / bbox_h
                        bbox_aspect_ratios.append(aspect_ratio)
                    
                    bbox_area = bbox_w * bbox_h
                    bbox_areas.append(bbox_area)
    
    return {
        'class_counts': class_counts,
        'image_sizes': image_sizes,
        'bbox_aspect_ratios': bbox_aspect_ratios,
        'bbox_areas': bbox_areas
    }

def visualize_random_images(images_dir, labels_dir, num_images=5):
    """Rastgele görüntüler seçer ve etiketleriyle birlikte görselleştirir"""
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    sampled_images = random.sample(image_files, min(num_images, len(image_files)))
    
    plt.figure(figsize=(15, 4 * num_images))
    
    for i, img_file in enumerate(sampled_images):
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
        
        # Görüntüyü oku
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Etiketleri oku
        labels = load_labels(label_path)
        
        plt.subplot(num_images, 1, i+1)
        plt.imshow(img)
        plt.title(f"Örnek Görüntü: {img_file}")
        
        # Etiketi olanları görüntü üzerinde göster
        for label in labels:
            class_id, x_center, y_center, width, height = label
            class_id = int(class_id)
            
            # Normalize edilmiş koordinatları piksel koordinatlarına dönüştür
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # Sınıfa göre renk belirleme
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
            color = colors[class_id % len(colors)]
            
            # Bounding box çizme
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Sınıf adını yazma
            cv2.putText(img, 
                        class_names.get(class_id, f"Class {class_id}"), 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2)
        
        plt.imshow(img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_ROOT, 'sample_images.png'))
    plt.close()

def plot_statistics(stats):
    """Veri seti istatistiklerini görselleştirir"""
    plt.figure(figsize=(18, 10))
    
    # Sınıf dağılımı
    plt.subplot(2, 2, 1)
    classes = [class_names.get(cls_id, f"Class {cls_id}") for cls_id in stats['class_counts'].keys()]
    counts = list(stats['class_counts'].values())
    plt.bar(classes, counts)
    plt.title('Sınıf Dağılımı')
    plt.xticks(rotation=45)
    plt.ylabel('Örnek Sayısı')
    
    # Görüntü boyutları
    plt.subplot(2, 2, 2)
    widths, heights = zip(*stats['image_sizes'])
    plt.scatter(widths, heights, alpha=0.5)
    plt.title('Görüntü Boyutları')
    plt.xlabel('Genişlik (piksel)')
    plt.ylabel('Yükseklik (piksel)')
    
    # Bounding Box En-Boy Oranları
    plt.subplot(2, 2, 3)
    plt.hist(stats['bbox_aspect_ratios'], bins=20, alpha=0.7)
    plt.title('Bounding Box En-Boy Oranları')
    plt.xlabel('En-Boy Oranı (Genişlik/Yükseklik)')
    plt.ylabel('Frekans')
    
    # Bounding Box Alanları
    plt.subplot(2, 2, 4)
    plt.hist(stats['bbox_areas'], bins=20, alpha=0.7)
    plt.title('Bounding Box Alanları')
    plt.xlabel('Alan (piksel²)')
    plt.ylabel('Frekans')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_ROOT, 'dataset_statistics.png'))
    plt.close()

def main():
    """Ana işlev - veri seti analizini çalıştırır"""
    print("Hit-UAV veri seti analizi başlıyor...")
    
    # Eğitim veri setini analiz et
    print("\nEğitim veri seti analizi:")
    train_stats = analyze_dataset(IMAGES_TRAIN, LABELS_TRAIN)
    
    # İstatistikleri görselleştir
    plot_statistics(train_stats)
    
    # Rastgele görüntüleri görselleştir
    print("\nÖrnek görüntüler hazırlanıyor...")
    visualize_random_images(IMAGES_TRAIN, LABELS_TRAIN, num_images=5)
    
    print("\nAnaliz tamamlandı. Görselleştirmeler kaydedildi.")
    print(f"Örnek görüntüler: {os.path.join(DATA_ROOT, 'sample_images.png')}")
    print(f"Veri seti istatistikleri: {os.path.join(DATA_ROOT, 'dataset_statistics.png')}")

if __name__ == "__main__":
    main()
