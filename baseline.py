import cv2
import numpy as np
import time
from utils import calculate_radial_distances

class BaselineDetector:
    def __init__(self):
        self.prev_frame_gray = None
        
        # Lab color range for "orange/fire"
        # We target pixels with high a (red/green) and high b (yellow/blue) channels
        self.lower_lab = np.array([0, 135, 135], dtype=np.uint8)
        self.upper_lab = np.array([255, 255, 255], dtype=np.uint8)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Frame Differencing
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return frame, {"contour_count": 0, "spread_metric": 0.0}
        
        frame_diff = cv2.absdiff(gray, self.prev_frame_gray)
        # Static threshold of 20
        _, motion_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
        self.prev_frame_gray = gray
        
        # 2. Lab Color Space Thresholding
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        color_mask = cv2.inRange(lab, self.lower_lab, self.upper_lab)
        
        # 3. Logical AND to isolate moving fire
        fire_mask = cv2.bitwise_and(motion_mask, color_mask)
        
        # 4. Contour extraction & Radial Distances
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vis_frame = frame.copy()
        
        contour_count = len(contours)
        all_variances = []
        
        for contour in contours:
            if cv2.contourArea(contour) < 200: # Filter small noise
                continue
                
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate 12 radial distances at 30-degree intervals
                distances = calculate_radial_distances(contour, (cx, cy), num_rays=12)
                all_variances.append(np.var(distances))
                
                # Visualizations
                cv2.drawContours(vis_frame, [contour], -1, (0, 0, 255), 2)
                cv2.circle(vis_frame, (cx, cy), 4, (0, 255, 0), -1)
                
                # Draw rays based on distances
                angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
                for i, angle in enumerate(angles):
                    d = distances[i]
                    if d > 0:
                        pt_x = int(cx + d * np.cos(angle))
                        pt_y = int(cy + d * np.sin(angle))
                        cv2.line(vis_frame, (cx, cy), (pt_x, pt_y), (255, 0, 0), 1)
        
        spread_metric = float(np.mean(all_variances)) if all_variances else 0.0
        metrics = {"contour_count": contour_count, "spread_metric": spread_metric}
        return vis_frame, metrics

def run_baseline(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    detector = BaselineDetector()
    print("Running Baseline Pipeline...")
    print("Press 'ESC' to exit the video window.")
    
    cv2.namedWindow("Classical CV Baseline - Fire Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Classical CV Baseline - Fire Detection", 1280, 720)
    
    frame_metrics = []
    frame_counter = 0
    
    # Setup VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_source = cap.get(cv2.CAP_PROP_FPS)
    if not fps_source or fps_source <= 0:
        fps_source = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter("baseline_output.mp4", fourcc, fps_source, (frame_width, frame_height))
    
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
            cv2.imwrite(f"baseline_frame_{frame_counter}.jpg", result_frame)
        
        cv2.imshow("Classical CV Baseline - Fire Detection", result_frame)
        
        if cv2.waitKey(30) & 0xFF == 27: # ESC to exit
            break
            
        if frame_counter >= 300:
            print("Reached exactly 300 frames. Early stopping for file size optimization.")
            break
            
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    return frame_metrics
