"""
UAV (İnsansız Hava Aracı) Güvenlik Sistemi için Optimizasyonlu Nesne Tanıma Modeli
-------------------------------------------------------------------------------------
Bu script, eğitilmiş YOLOv8 modelini kullanarak gerçek zamanlı nesne tespiti yapar ve
güvenlikle ilgili bilgileri rapor eder. Drone sistemlerinde çalışacak şekilde optimize edilmiştir.

Kullanım:
    python deploy_security_model.py --model model_results/security_model_n_v1/weights/best.pt --source 0
"""
from ultralytics import YOLO
import os
import cv2
import numpy as np
import argparse
import time
import torch
import logging
import shutil
from typing import List, Dict, Tuple, Union, Optional
import threading
import sys

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("uav_security.log"),
        logging.StreamHandler()
    ]
)

# Projenin kök dizini
PROJECT_ROOT = r'c:\Users\Casper\Desktop\UAV'
MODEL_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'model_results')

# Sınıf adları
class_names = {
    0: 'Person',
    1: 'Car',
    2: 'Bicycle',
    3: 'OtherVehicle',
    4: 'DontCare'
}

# Güvenlik analitiği için sınıflandırmalar
security_classes = {
    0: 'high',  # İnsan - yüksek güvenlik önceliği
    1: 'medium',  # Araba - orta güvenlik önceliği
    2: 'medium',  # Bisiklet - orta güvenlik önceliği
    3: 'medium',  # Diğer Araçlar - orta güvenlik önceliği
    4: 'low'  # Önemsiz - düşük güvenlik önceliği
}

def optimize_model(model_path: str, format_type: str = 'onnx') -> str:
    """
    Modeli belirtilen formatta optimize eder
    
    Args:
        model_path: Orijinal model dosya yolu
        format_type: 'onnx' veya 'trt' (TensorRT)
    
    Returns:
        Optimize edilmiş model yolu
    """
    try:
        model = YOLO(model_path)
        
        # Model adını al
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Optimize edilmiş model dizini
        export_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'optimized')
        os.makedirs(export_dir, exist_ok=True)
        
        if format_type.lower() == 'onnx':
            logging.info(f"Model ONNX formatına optimize ediliyor...")
            # ONNX formatına dönüştür
            model.export(format='onnx', dynamic=True, simplify=True)
            optimized_path = os.path.join(export_dir, f"{model_name}.onnx")
            
            if os.path.exists(f"{model_name}.onnx"):
                shutil.move(f"{model_name}.onnx", optimized_path)
                logging.info(f"Model başarıyla optimize edildi: {optimized_path}")
                return optimized_path
        
        elif format_type.lower() == 'trt' and torch.cuda.is_available():
            logging.info(f"Model TensorRT formatına optimize ediliyor...")
            # TensorRT formatına dönüştür
            model.export(format='engine', dynamic=True, simplify=True, workspace=4)
            optimized_path = os.path.join(export_dir, f"{model_name}.engine")
            
            if os.path.exists(f"{model_name}.engine"):
                shutil.move(f"{model_name}.engine", optimized_path)
                logging.info(f"Model başarıyla optimize edildi: {optimized_path}")
                return optimized_path
        
        logging.warning(f"Model optimizasyonu başarısız oldu, orijinal model kullanılacak")
        return model_path
    
    except Exception as e:
        logging.error(f"Model optimizasyon hatası: {e}")
        return model_path

def parse_arguments():
    """Komut satırı argümanlarını işler"""
    parser = argparse.ArgumentParser(description='UAV Güvenlik Sistemleri için Nesne Tanıma')
    parser.add_argument('--model', type=str, 
                      default='model_results/security_model_n_v1/weights/best.pt',
                      help='Eğitilmiş model yolu')
    parser.add_argument('--source', type=str, default='0',
                      help='Görüntü/video kaynağı. "0" webcam için, video dosyası için dosya yolu')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Minimum tespit güven eşiği')
    parser.add_argument('--iou', type=float, default=0.45,
                      help='IOU eşiği (Non-maximum suppression için)')
    parser.add_argument('--device', type=str, default='',
                      help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--output', type=str, default='output.mp4',
                      help='Çıktı video dosyası (sadece video kaynağı için)')
    parser.add_argument('--save_frames', action='store_true',
                      help='Önemli güvenlik tespitleri için kareleri kaydet')
    parser.add_argument('--optimize', action='store_true',
                      help='Modeli TensorRT veya ONNX ile optimize et')
    parser.add_argument('--view_img', action='store_true',
                      help='Tespit sonuçlarını görüntüle')
    parser.add_argument('--low_memory', action='store_true',
                      help='Düşük bellek kullanımı modu (UAV için)')
    parser.add_argument('--imgsz', type=int, default=640,
                      help='Çıkarım için görüntü boyutu')
    parser.add_argument('--half', action='store_true',
                      help='FP16 yarı hassasiyette çıkarım kullan')
    return parser.parse_args()

def security_alert_level(detections):
    """Tespitlere dayalı genel güvenlik uyarı seviyesini belirler"""
    if not detections:
        return "LOW", "Normal durum"
    
    # Öncelik sayaçları
    priority_counts = {
        'high': 0,
        'medium': 0,
        'low': 0
    }
    
    # Tespit edilen sınıf sayıları
    class_counts = {}
    
    for det in detections:
        class_id = int(det[5])
        priority = security_classes.get(class_id, 'low')
        priority_counts[priority] += 1
        
        class_name = class_names.get(class_id, f"Class {class_id}")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Güvenlik seviyesini belirle
    if priority_counts['high'] > 0:
        alert_level = "HIGH"
        
        # Birden fazla kişi varsa ekstra bilgi ekle
        if priority_counts['high'] > 1:
            message = f"DİKKAT: {priority_counts['high']} kişi tespit edildi!"
        else:
            message = "DİKKAT: Kişi tespit edildi!"
            
    elif priority_counts['medium'] > 0:
        alert_level = "MEDIUM"
        
        # Araç tipleri hakkında bilgi ver
        vehicle_types = []
        for cls, count in class_counts.items():
            if cls in ['Car', 'Bicycle', 'OtherVehicle']:
                vehicle_types.append(f"{count} {cls}")
        
        message = f"Uyarı: {', '.join(vehicle_types)} tespit edildi"
    else:
        alert_level = "LOW"
        message = "Normal durum"
    
    return alert_level, message

def frame_preprocessor(frame: np.ndarray, imgsz: int = 640, low_memory: bool = False) -> np.ndarray:
    """
    UAV için kare önişleme optimizasyonları
    
    Args:
        frame: İşlenecek görüntü
        imgsz: Hedef görüntü boyutu
        low_memory: Düşük bellek modu açık/kapalı
    
    Returns:
        Önişlenmiş görüntü
    """
    # Düşük bellek modunda görüntü boyutunu daha da küçült
    if low_memory:
        imgsz = min(imgsz, 416)  # Daha küçük bir görüntü boyutu kullan
    
    # Orjinal boyutları sakla
    h, w = frame.shape[:2]
    
    # Görüntüyü belirtilen boyuta yeniden boyutlandır
    if h > imgsz or w > imgsz:
        # En-boy oranını koru
        scale = imgsz / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return frame

def draw_security_overlay(frame: np.ndarray, detections: List, fps: float) -> np.ndarray:
    """Güvenlik bilgilerini içeren overlay'i çiz"""
    h, w = frame.shape[:2]
    
    # Güvenlik seviyesini belirle
    alert_level, message = security_alert_level(detections)
    
    # Yarı saydam üst bilgi paneli çiz
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # FPS ve tarih/saat bilgisi ekle
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"FPS: {fps:.1f} | {current_time}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Güvenlik durumu ekle
    alert_color = {
        "HIGH": (0, 0, 255),   # Kırmızı - Yüksek öncelik
        "MEDIUM": (0, 165, 255),  # Turuncu - Orta öncelik
        "LOW": (0, 255, 0)     # Yeşil - Düşük öncelik
    }
    
    color = alert_color.get(alert_level, (200, 200, 200))
    cv2.putText(frame, f"GÜVENLİK SEVİYESİ: {alert_level}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, message, (w//2, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Ekstra drone bilgilerini ekle - gerçek bir UAV entegrasyonunda bu değerler drone API'sinden alınabilir
    battery_level = 85  # Örnek değer
    altitude = 25.4     # Örnek değer (metre)
    speed = 12.6        # Örnek değer (m/s)
    
    cv2.putText(frame, f"Batarya: %{battery_level} | Yükseklik: {altitude}m | Hız: {speed}m/s", 
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def run_security_detection(
    model_path: str,
    source: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    output_path: Optional[str] = None,
    save_frames: bool = False,
    view_img: bool = True,
    device: str = '',
    low_memory: bool = False,
    imgsz: int = 640,
    half: bool = False
):
    """Güvenlik amaçlı nesne tespiti çalıştırır"""
    # Cihazı ayarla
    if device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Model yükleniyor: {model_path}")
    logging.info(f"Cihaz: {device}")
    
    # Modeli yükle
    try:
        model = YOLO(model_path)
        
        # Çıkarım için gereken ayarları yap
        model.to(device)
        
        if half and device != 'cpu':
            model.model.half()  # FP16 yarı-hassasiyet
            logging.info("FP16 yarı-hassasiyet modu aktif")
    except Exception as e:
        logging.error(f"Model yükleme hatası: {e}")
        return
    
    # Video kaynağı ayarla
    is_webcam = source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
    try:
        cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)
    except Exception as e:
        logging.error(f"Kamera bağlantı hatası: {e}")
        return
    
    if not cap.isOpened():
        logging.error(f"Hata: Video kaynağı açılamadı: {source}")
        return
    
    # Video parametrelerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logging.info(f"Video boyutu: {width}x{height}, FPS: {fps:.1f}")
    
    # Çıktı videosu ayarları
    out = None
    if output_path and not is_webcam:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        logging.info(f"Çıktı videosu: {output_path}")
    
    # Kare sayacı ve FPS hesaplaması için değişkenler
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    # Güvenlik uyarıları için dizin oluştur
    if save_frames:
        save_dir = os.path.join(PROJECT_ROOT, "security_alerts")
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Güvenlik uyarıları için resimler {save_dir} dizinine kaydedilecek")
    
    # Bellek tüketimini izlemek için thread başlat (eğer gerekiyorsa)
    if low_memory:
        logging.info("Düşük bellek modu aktif - performans optimizasyonları uygulanıyor")
    
    logging.info("Nesne tespiti başlatılıyor...")
    
    # Her karede işlem yapmak yerine kare atlama 
    skip_frames = 2 if low_memory else 0  # Düşük bellek modunda her 3 kareden birini işle
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if not is_webcam:
                    logging.info("Video işleme tamamlandı.")
                break
            
            frame_count += 1
            
            # Düşük bellek modunda kare atlama
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                if view_img:
                    cv2.imshow("UAV Güvenlik Sistemi", frame)
                if out:
                    out.write(frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue
            
            # FPS hesapla
            current_time = time.time()
            if frame_count % 10 == 0:
                fps_display = 10 / (current_time - start_time)
                start_time = current_time
            
            # Kareyi önişle (optimize et)
            processed_frame = frame_preprocessor(frame, imgsz, low_memory)
            
            # Model ile nesne tespiti yap
            try:
                results = model.predict(
                    processed_frame, 
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )[0]
            except Exception as e:
                logging.error(f"Tahmin hatası: {e}")
                continue
            
            # Tespitleri işle
            detections = []
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                for det in results.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, confidence, class_id = det
                    
                    # Orijinal çözünürlüğe ölçekle
                    if processed_frame.shape[:2] != frame.shape[:2]:
                        h_ratio = frame.shape[0] / processed_frame.shape[0]
                        w_ratio = frame.shape[1] / processed_frame.shape[1]
                        x1, x2 = x1 * w_ratio, x2 * w_ratio
                        y1, y2 = y1 * h_ratio, y2 * h_ratio
                    
                    detections.append([x1, y1, x2, y2, confidence, class_id])
                    
                    # Bounding box çiz
                    class_id = int(class_id)
                    priority = security_classes.get(class_id, 'low')
                    
                    # Güvenlik önceliğine göre renk belirleme
                    color = {
                        'high': (0, 0, 255),    # Kırmızı - Yüksek öncelik
                        'medium': (0, 165, 255),  # Turuncu - Orta öncelik
                        'low': (0, 255, 0)      # Yeşil - Düşük öncelik
                    }.get(priority, (255, 255, 255))
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Etiket ekle
                    label = f"{class_names.get(class_id, 'Unknown')} {confidence:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Güvenlik overlay'ini ekle
            frame = draw_security_overlay(frame, detections, fps_display)
              # Yüksek öncelikli güvenlik uyarısı varsa kareyi kaydet
            if save_frames and detections and any(security_classes.get(int(det[5]), 'low') == 'high' for det in detections):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join("security_alerts", f"alert_{timestamp}_{frame_count}.jpg")
                cv2.imwrite(save_path, frame)
                logging.info(f"Güvenlik uyarısı! Resim kaydedildi: {save_path}")
              # Görüntüyü göster - şimdilik devre dışı bırakıldı
            # if view_img:
            #    cv2.imshow("UAV Güvenlik Sistemi", frame)
            
            # Çıktı videosuna yaz
            if out:
                out.write(frame)
              # 'q' tuşu ile çıkış yap - şimdilik devre dışı bırakıldı
            # if cv2.waitKey(1) == ord('q'):
            #    break
            
            # Bellek yönetimi
            if low_memory and frame_count % 30 == 0:
                # Bellek temizleme
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    except KeyboardInterrupt:
        logging.info("Kullanıcı tarafından durduruldu")
    except Exception as e:
        logging.error(f"Hata: {e}")
    finally:        # Kaynakları serbest bırak
        cap.release()
        if out:
            out.release()
        # cv2.destroyAllWindows()  # şimdilik devre dışı bırakıldı
        
        logging.info("Nesne tespiti tamamlandı.")

def main():
    """Ana işlev"""
    print("\n" + "="*70)
    print("          UAV GÜVENLİK SİSTEMİ - YOLOv8 NESNE TESPİTİ")
    print("="*70)
    
    # Komut satırı argümanlarını al
    args = parse_arguments()
    
    # Model yolunu belirle
    model_path = os.path.join(PROJECT_ROOT, args.model)
    if not os.path.exists(model_path):
        logging.error(f"Model dosyası bulunamadı: {model_path}")
        return
    
    # Sistem bilgilerini görüntüle
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU: {device_name}")
        print(f"CUDA Sürümü: {torch.version.cuda}")
    
    print(f"\nSistem Bilgisi:")
    print(f"- İşlemci: {device_info}")
    print(f"- PyTorch Sürümü: {torch.__version__}")
    print(f"- OpenCV Sürümü: {cv2.__version__}")
    
    print(f"\nModel Bilgisi:")
    print(f"- Model: {os.path.basename(model_path)}")
    print(f"- Kaynak: {'Kamera' if args.source == '0' else args.source}")
    print(f"- Görüntü boyutu: {args.imgsz}x{args.imgsz}")
    print(f"- Güven eşiği: {args.conf}")
    
    # Model optimizasyonu
    if args.optimize:
        if torch.cuda.is_available():
            print("\nModel TensorRT formatına optimize ediliyor...")
            model_path = optimize_model(model_path, 'trt')
        else:
            print("\nModel ONNX formatına optimize ediliyor...")
            model_path = optimize_model(model_path, 'onnx')
    
    # Onay iste
    if not args.source.isnumeric():
        start_detection = input("\nTespit başlatılsın mı? [E/h]: ").lower() != 'h'
        if not start_detection:
            print("Tespit iptal edildi.")
            return
    
    print("\nNesne tespiti başlatılıyor...\n")
    
    # Tespit işlemini başlat
    run_security_detection(
        model_path=model_path,
        source=args.source,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_path=args.output if args.source != '0' else None,
        save_frames=args.save_frames,
        view_img=args.view_img,
        device=args.device,
        low_memory=args.low_memory,
        imgsz=args.imgsz,
        half=args.half
    )

if __name__ == "__main__":
    main()
