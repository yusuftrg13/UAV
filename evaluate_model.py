from ultralytics import YOLO
import numpy as np
import os
import sys

def evaluate_model_on_test_set():
    """Test setindeki tüm görüntülerde nesne tespiti yapar ve sonuçları analiz eder"""
    print("\n==== UAV GÜVENLİK MODELİ PERFORMANS ANALİZİ ====\n")
    sys.stdout.flush() # Çıktının hemen görünmesini sağla
    
    # Modeli yükle
    model_path = 'model_results/security_model_n_v1/weights/best.pt'
    model = YOLO(model_path)
    print(f"Model: {os.path.basename(model_path)}")
    
    # Test klasörü
    test_dir = 'hit-uav/images/test'
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]
    print(f"Test görüntü sayısı: {len(test_images)}")
    
    # Görüntülerde tahmin yap
    results = model.predict(source=test_dir, save=False, conf=0.25)
    print("Tahminler tamamlandı")
    
    # Güven değerlerini topla
    confidences = []
    class_counts = {}
    
    for r in results:
        if len(r.boxes) > 0:
            confidences.extend(r.boxes.conf.cpu().numpy())
            
            for i, c in enumerate(r.boxes.cls.cpu().numpy()):
                class_id = int(c)
                name = model.names[class_id]
                conf = float(r.boxes.conf[i])
                
                if name not in class_counts:
                    class_counts[name] = {"count": 0, "conf_sum": 0}
                
                class_counts[name]["count"] += 1
                class_counts[name]["conf_sum"] += conf
    
    # Sonuçları analiz et
    print("\n----- GENEL İSTATİSTİKLER -----")
    if confidences:
        print(f"Toplam tespit edilen nesne sayısı: {len(confidences)}")
        print(f"Ortalama tespit güveni: {np.mean(confidences):.4f}")
        print(f"Minimum tespit güveni: {np.min(confidences):.4f}")
        print(f"Maksimum tespit güveni: {np.max(confidences):.4f}")
    else:
        print("Hiç nesne tespit edilemedi!")
    
    # Sınıf bazlı analiz
    print("\n----- SINIF BAZLI TESPİT İSTATİSTİKLERİ -----")
    for name, stats in class_counts.items():
        count = stats["count"]
        avg_conf = stats["conf_sum"] / count if count > 0 else 0
        print(f"- {name}: {count} adet (Ortalama güven: {avg_conf:.4f})")
    
    # Tespit edilen nesne sayısı olan görüntü sayısı
    images_with_detections = sum(1 for r in results if len(r.boxes) > 0)
    empty_images = len(results) - images_with_detections
    
    print(f"\nNesne tespit edilen görüntü sayısı: {images_with_detections} ({images_with_detections/len(results)*100:.2f}%)")
    print(f"Nesne tespit edilemeyen görüntü sayısı: {empty_images} ({empty_images/len(results)*100:.2f}%)")
    
    print("\n==== DEĞERLENDİRME TAMAMLANDI ====")

if __name__ == "__main__":
    evaluate_model_on_test_set()
