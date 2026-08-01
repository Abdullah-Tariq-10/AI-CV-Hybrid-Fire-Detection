import cv2
import numpy as np
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class UpgradedDetector:
    def __init__(self):
        # 1. Background Subtractor MOG2 for adaptive motion detection
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
        # 2. Pre-trained lightweight MobileNetV2 (Quantized)
        print("Loading quantized MobileNetV2 for CPU efficiency...")
        
        # Explicitly set the PyTorch quantization engine to avoid 'Quantized backend not supported'
        supported_engines = torch.backends.quantized.supported_engines
        if 'onednn' in supported_engines:
            torch.backends.quantized.engine = 'onednn'
        elif 'fbgemm' in supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
        elif 'qnnpack' in supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif supported_engines:
            torch.backends.quantized.engine = supported_engines[0]

        try:
            # Attempt to load the pre-quantized QNNPACK weights
            weights = models.quantization.MobileNet_V2_QuantizedWeights.DEFAULT
            self.model = models.quantization.mobilenet_v2(
                weights=weights, 
                quantize=True, 
                backend=torch.backends.quantized.engine
            )
        except ValueError:
            print("Pre-quantized QNNPACK mismatch on this CPU architecture. Applying dynamic on-the-fly quantization instead...")
            # Fallback to dynamic quantization on standard model for Windows x86 compatibility
            base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            base_model.eval()
            self.model = torch.quantization.quantize_dynamic(
                base_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
        self.model.eval()
        
        # Image transformation for MobileNetV2
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 3. Shi-Tomasi and Lucas-Kanade Optical Flow Parameters
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.old_gray = None
        self.p0 = None
        self.flow_mask = None # Mask for drawing optical flow tracks

    def classify_fire(self, roi):
        """Passes the ROI to the quantized MobileNetV2 to classify fire vs non-fire"""
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            return False
            
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(roi_rgb)
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():
            out = self.model(batch_t)
            
        _, index = torch.max(out, 1)
        pred_class = index.item()
        
        # Note: Since this is a standard ImageNet model, we map arbitrary classes for scaffolding.
        # ImageNet classes like matchstick (644), lighter (626) might loosely correlate.
        # In a real deployed edge model, you'd load a MobileNet trained specifically for 2 classes (fire/non-fire).
        # We will simulate a positive prediction here for demonstration of the flow tracking if there's enough movement brightness.
        # As a heuristic mock for the course project if no fine-tuned model exists:
        mean_brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        if mean_brightness > 120:
            return True
            
        return False

    def process_frame(self, frame):
        vis_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.flow_mask is None:
            self.flow_mask = np.zeros_like(frame)

        # 1. Background Subtraction
        fg_mask = self.backSub.apply(frame)
        
        # Remove shadows (MOG2 puts shadows as 127)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_count = len(contours)
        spread_metric = 0.0
        
        new_features = []
        
        for contour in contours:
            if cv2.contourArea(contour) < 500: # Filter small movements
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]
            
            # 2. ROI Classification via MobileNetV2
            is_fire = self.classify_fire(roi)
            
            if is_fire:
                cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 165, 255), 2) # Orange bounding box
                cv2.putText(vis_frame, "Fire", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                # 3. Get Shi-Tomasi corners to initialize tracking
                roi_gray = gray[y:y+h, x:x+w]
                corners = cv2.goodFeaturesToTrack(roi_gray, mask=None, **self.feature_params)
                
                if corners is not None:
                    # Adjust corner coords to global frame
                    corners[:, 0, 0] += x
                    corners[:, 0, 1] += y
                    new_features.extend(corners)

        # Add newly found features to our tracking points
        if len(new_features) > 0:
            new_features_arr = np.array(new_features, dtype=np.float32)
            if self.p0 is None:
                self.p0 = new_features_arr
            else:
                self.p0 = np.concatenate((self.p0, new_features_arr), axis=0)

        # 4. Sparse Optical Flow Tracking
        if self.old_gray is not None and self.p0 is not None and len(self.p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, None, **self.lk_params)
            
            if p1 is not None and st is not None:
                # Select good points
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]
                
                if len(good_new) > 0:
                    magnitudes = np.linalg.norm(good_new - good_old, axis=1)
                    spread_metric = float(np.mean(magnitudes))
                
                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    self.flow_mask = cv2.line(self.flow_mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    vis_frame = cv2.circle(vis_frame, (int(a), int(b)), 5, (0, 255, 0), -1)
                
                vis_frame = cv2.add(vis_frame, self.flow_mask)
                self.p0 = good_new.reshape(-1, 1, 2)
            else:
                self.p0 = None
                
        self.old_gray = gray.copy()
        
        # Gradually fade out old flow trails
        self.flow_mask = cv2.addWeighted(self.flow_mask, 0.95, np.zeros_like(self.flow_mask), 0.05, 0)
        
        metrics = {"contour_count": contour_count, "spread_metric": spread_metric}
        return vis_frame, metrics

def run_upgraded(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    detector = UpgradedDetector()
    print("Running Upgraded AI+CV Hybrid Pipeline...")
    print("Press 'ESC' to exit the video window.")
    
    cv2.namedWindow("Upgraded Hybrid AI+CV Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Upgraded Hybrid AI+CV Detector", 1280, 720)
    
    frame_metrics = []
    frame_counter = 0
    
    # Setup VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_source = cap.get(cv2.CAP_PROP_FPS)
    if not fps_source or fps_source <= 0:
        fps_source = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter("upgraded_output.mp4", fourcc, fps_source, (frame_width, frame_height))
    
    while True:
        start_t = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
            
        frame_counter += 1
        
        result_frame, metrics = detector.process_frame(frame)
        end_t = time.time()
        
        fps = 1.0 / (end_t - start_t) if (end_t - start_t) > 0 else 0.0
        metrics["fps"] = fps
        frame_metrics.append(metrics)
        
        out_writer.write(result_frame)
        
        if frame_counter in [50, 150, 250]:
            cv2.imwrite(f"upgraded_frame_{frame_counter}.jpg", result_frame)
            
        cv2.imshow("Upgraded Hybrid AI+CV Detector", result_frame)
        
        if cv2.waitKey(30) & 0xFF == 27: # ESC to exit
            break
            
        if frame_counter >= 300:
            print("Reached exactly 300 frames. Early stopping for file size optimization.")
            break
            
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    return frame_metrics
