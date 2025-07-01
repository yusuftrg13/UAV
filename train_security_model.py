from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import shutil
import torch
import sys

# Projenin kök dizini
PROJECT_ROOT = r'c:\Users\Casper\Desktop\UAV'
DATA_PATH = os.path.join(PROJECT_ROOT, 'hit-uav', 'config.yaml')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'model_results')

# Sonuçlar dizini yoksa oluştur
os.makedirs(RESULTS_DIR, exist_ok=True)

def check_model_file(model_size):
    """
    Model dosyasının varlığını kontrol eder ve yoksa indirmeyi dener
    
    Args:
        model_size (str): Model büyüklüğü (n, s, m, l, x)
    
    Returns:
        bool: Model dosyası mevcutsa True, değilse False
    """
    model_file = f"yolov8{model_size}.pt"
    if os.path.exists(model_file):
        print(f"{model_file} dosyası mevcut.")
        return True
    
    print(f"{model_file} dosyası bulunamadı. İndirme işlemi başlatılıyor...")
    try:
        # YOLO, model dosyasını otomatik indirecektir
        model = YOLO(model_file)
        print(f"{model_file} başarıyla indirildi.")
        return True
    except Exception as e:
        print(f"Model indirme hatası: {e}")
        return False

def train_security_model(model_size='s', epochs=20, imgsz=640, batch=16, name='security_model'):
    """
    YOLOv8 modeli eğitir
    
    Args:
        model_size (str): Model büyüklüğü (n, s, m, l, x)
        epochs (int): Eğitim için epoch sayısı
        imgsz (int): Görüntü boyutu
        batch (int): Batch size
        name (str): Model ismi
    """
    print(f"YOLOv8{model_size} modeli eğitimi başlatılıyor...")
    
    # Model dosyasının varlığını kontrol et
    if not check_model_file(model_size):
        print("Model dosyası bulunamadı. Eğitim iptal ediliyor.")
        return None
    
    # YOLOv8 modelini yükle
    model = YOLO(f"yolov8{model_size}.pt")
    
    # GPU kullanılabilirliğini kontrol et
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Eğitim için {device} kullanılıyor")
    
    # Model eğitimi
    results = model.train(
        data=DATA_PATH,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        patience=7,  # Early stopping için sabır parametresi
        project=RESULTS_DIR,  # Sonuçların kaydedileceği dizin
        pretrained=True,  # Önceden eğitilmiş ağırlıkları kullan
        optimizer='auto',  # Otomatik optimizer seçimi
        cache=True,  # Verileri önbellekte tut (daha hızlı eğitim)
        device=device,
        amp=True,  # Mixed precision training
        workers=4,  # Veri yükleme iş parçacığı sayısı
        close_mosaic=10,  # Son 10 epoch'ta mozaik augmentation'ı kapat
        cos_lr=True  # Cosine learning rate scheduler
    )
    
    print(f"Eğitim tamamlandı! Sonuçlar {RESULTS_DIR}/{name} dizininde kaydedildi.")
    return results

def evaluate_model(model_path):
    """
    Eğitilmiş modeli değerlendirir
    
    Args:
        model_path (str): Eğitilmiş model yolu
    """
    print(f"Model değerlendirmesi başlatılıyor: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return None
    
    # Modeli yükle
    model = YOLO(model_path)
    
    # Test veri setinde değerlendirme yap
    results = model.val(data=DATA_PATH)
    
    try:
        # Metrikler
        print("\n==== Model Performans Metrikleri ====")
        
        # Genel metrikler - TypeError hatası için float() ile dönüştürme
        print(f"mAP50: {float(results.box.map50):.4f}")
        print(f"mAP50-95: {float(results.box.map):.4f}")
        print(f"Precision: {float(results.box.p):.4f}")
        print(f"Recall: {float(results.box.r):.4f}")
        print(f"F1-Score: {float(results.box.f1):.4f}")
        
        # Sınıf bazlı metrikleri ekrana yazdır
        if hasattr(results, 'names') and hasattr(results.box, 'ap_class_index'):
            print("\n==== Sınıf Bazlı Performans ====")
            class_indices = results.box.ap_class_index
            for idx in class_indices:
                class_name = results.names[int(idx)]
                class_ap50 = float(results.box.ap50[int(idx)])
                print(f"{class_name}: mAP50 = {class_ap50:.4f}")
    except Exception as e:
        print(f"Değerlendirme sonuçlarını görüntüleme hatası: {e}")
        print("Ham değerlendirme sonuçları:", results)
    
    return results

def test_on_images(model_path, test_image_dir):
    """
    Eğitilmiş modeli test görüntüleri üzerinde dener
    
    Args:
        model_path (str): Eğitilmiş model yolu
        test_image_dir (str): Test görüntülerinin bulunduğu dizin
    """
    print(f"Model test ediliyor: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return None
    
    if not os.path.exists(test_image_dir):
        print(f"HATA: Test görüntü dizini bulunamadı: {test_image_dir}")
        return None
    
    # Modeli yükle
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None
    
    # Test dizinindeki görüntü sayısını kontrol et
    test_files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not test_files:
        print(f"Test dizininde görüntü dosyası bulunamadı: {test_image_dir}")
        return None
    
    print(f"Toplam {len(test_files)} test görüntüsü işlenecek...")
    
    # Tahminler için dizin
    prediction_dir = os.path.join(RESULTS_DIR, 'predictions')
    os.makedirs(prediction_dir, exist_ok=True)
    
    # Test görüntüleri üzerinde tahmin yap
    try:
        results = model.predict(
            source=test_image_dir,
            save=True,
            conf=0.25,  # Güven eşiği
            iou=0.45,   # IoU eşiği
            max_det=100,  # Görüntü başına maksimum tespit sayısı
            project=prediction_dir,
            name=os.path.basename(os.path.dirname(model_path)),
            save_txt=True,  # Tespit sonuçlarını metin olarak da kaydet
            save_conf=True,  # Güven değerlerini metin dosyalarına kaydet
            line_width=2,    # Tespit kutularının çizgi kalınlığı
            boxes=True       # Tespit kutularını görselleştir
        )
        
        # Sonuçları göster
        print(f"\n==== Test Sonuçları ====")
        print(f"İşlenen görüntü sayısı: {len(results)}")
        print(f"Tahmin sonuçları kaydedildi: {os.path.join(prediction_dir, os.path.basename(os.path.dirname(model_path)))}")
        
        # Tespit edilen sınıf sayılarını hesapla
        if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            all_classes = []
            for r in results:
                if hasattr(r, 'boxes') and len(r.boxes) > 0:
                    all_classes.extend(r.boxes.cls.cpu().numpy())
            
            if all_classes:
                class_counts = {}
                for cls in all_classes:
                    cls_id = int(cls)
                    cls_name = results[0].names[cls_id] if cls_id in results[0].names else f"Class {cls_id}"
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                
                print("\nTespit edilen nesneler:")
                for cls_name, count in class_counts.items():
                    print(f"- {cls_name}: {count} adet")
        
        return results
    except Exception as e:
        print(f"Tahmin işlemi sırasında hata: {e}")
        return None

def download_model(model_size):
    """
    Eğer yoksa, belirtilen boyuttaki YOLOv8 modelini indirir
    
    Args:
        model_size (str): İndirilecek model boyutu (n, s, m, l, x)
    """
    model_file = f"yolov8{model_size}.pt"
    if not os.path.exists(model_file):
        print(f"Model {model_file} indiriliyor...")
        try:
            # Bu işlem modeli otomatik olarak indirecek
            YOLO(model_file)
            print(f"Model {model_file} başarıyla indirildi.")
        except Exception as e:
            print(f"Model indirme hatası: {e}")
            return False
    return True

def main():
    """Ana işlev"""
    print("\n" + "="*60)
    print("       GÜVENLİK AMAÇLI UAV NESNE TANIMA MODELİ EĞİTİMİ")
    print("="*60)
    
    # Sistem bilgilerini görüntüle
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Yok"
    print(f"\nSistem Bilgisi:")
    print(f"- Python versiyonu: {sys.version.split()[0]}")
    print(f"- PyTorch versiyonu: {torch.__version__}")
    print(f"- Eğitim cihazı: {device}")
    if torch.cuda.is_available():
        print(f"- GPU: {gpu_name}")
        print(f"- Kullanılabilir GPU belleği: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    print("\nVeri Yolu Kontrolü:")
    print(f"- Veri konfigürasyonu: {DATA_PATH}")
    print(f"- Sonuç dizini: {RESULTS_DIR}")
    
    # Model boyutunu seç (n=nano, s=small, m=medium)
    print("\nModel Seçenekleri:")
    print("- n: YOLOv8n (nano)  - En küçük ve en hızlı model, UAV'larda kullanım için uygundur")
    print("- s: YOLOv8s (small) - Daha doğru ama daha yavaş")
    print("- m: YOLOv8m (medium) - Daha doğru ancak daha fazla kaynak gerektiren model")
    
    model_size = input("\nHangi model boyutunu kullanmak istiyorsunuz? [n/s/m] (Varsayılan: n): ").lower() or 'n'
    if model_size not in ['n', 's', 'm']:
        print("Geçersiz seçim. Varsayılan 'n' (nano) model kullanılacak.")
        model_size = 'n'
    
    # Epoch sayısını seç
    try:
        epochs = int(input("\nKaç epoch eğitmek istiyorsunuz? (Varsayılan: 20): ") or 20)
        if epochs <= 0:
            print("Geçersiz epoch sayısı. Varsayılan 20 kullanılacak.")
            epochs = 20
    except ValueError:
        print("Geçersiz değer. Varsayılan 20 epoch kullanılacak.")
        epochs = 20
    
    # Batch size belirleme - GPU belleğine göre optimize et
    default_batch = 16
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem < 4:  # 4GB'dan az bellek
            default_batch = 8
        elif gpu_mem > 8:  # 8GB'dan fazla bellek
            default_batch = 32
    
    try:
        batch = int(input(f"\nBatch size kaç olsun? (Varsayılan: {default_batch}): ") or default_batch)
        if batch <= 0:
            print(f"Geçersiz batch size. Varsayılan {default_batch} kullanılacak.")
            batch = default_batch
    except ValueError:
        print(f"Geçersiz değer. Varsayılan {default_batch} kullanılacak.")
        batch = default_batch
    
    # Görüntü boyutu
    try:
        imgsz = int(input("\nGörüntü boyutu kaç olsun? (Varsayılan: 640): ") or 640)
        if imgsz <= 0:
            print("Geçersiz görüntü boyutu. Varsayılan 640 kullanılacak.")
            imgsz = 640
    except ValueError:
        print("Geçersiz değer. Varsayılan 640 kullanılacak.")
        imgsz = 640
    
    # Model ismi
    timestamp = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    default_name = f'security_model_{model_size}_v1'
    model_name = input(f"\nModel ismi ne olsun? (Varsayılan: {default_name}): ") or default_name
    
    # Eğitim parametrelerini özetle
    print("\n" + "="*60)
    print("EĞİTİM PARAMETRELERİ:")
    print(f"- Model: YOLOv8{model_size}")
    print(f"- Epoch sayısı: {epochs}")
    print(f"- Batch size: {batch}")
    print(f"- Görüntü boyutu: {imgsz}x{imgsz}")
    print(f"- Model ismi: {model_name}")
    print(f"- Eğitim cihazı: {device}")
    print("="*60 + "\n")
    
    start_train = input("Eğitime başlamak istiyor musunuz? [E/h]: ").lower() != 'h'
    if not start_train:
        print("Eğitim iptal edildi.")
        return
    
    print("\nEğitim başlatılıyor...\n")
    
    # 1. YOLOv8 modelini eğit
    results = train_security_model(
        model_size=model_size,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=model_name
    )
    
    if results is None:
        print("Eğitim sırasında bir hata oluştu. Program sonlandırılıyor.")
        return
    
    # 2. Eğitilen modeli değerlendir
    model_path = os.path.join(RESULTS_DIR, model_name, 'weights', 'best.pt')
    if os.path.exists(model_path):
        print("\nEğitilen model değerlendiriliyor...")
        evaluate_model(model_path)
        
        # 3. Test görüntüleri üzerinde dene
        test_dir = os.path.join(PROJECT_ROOT, 'hit-uav', 'images', 'test')
        if os.path.exists(test_dir):
            print("\nModel test görüntüleri üzerinde deneniyor...")
            test_on_images(model_path, test_dir)
        else:
            print(f"\nTest dizini bulunamadı: {test_dir}")
    else:
        print(f"\nEğitilen model dosyası bulunamadı: {model_path}")
    
    print("\n" + "="*60)
    print("Model eğitimi ve değerlendirmesi tamamlandı.")
    print(f"Sonuçlar '{os.path.join(RESULTS_DIR, model_name)}' dizinine kaydedildi.")
    print("="*60)

if __name__ == "__main__":
    main()
