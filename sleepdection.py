import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import threading
import queue
import time
import mediapipe as mp
from collections import deque


# Load OpenCV cascades with enhanced parameters
@st.cache_resource
def load_detectors():
    """Load OpenCV face and eye detectors with optimized parameters"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        return face_cascade, eye_cascade, profile_cascade
    except Exception as e:
        st.error(f"Error loading OpenCV cascades: {e}")
        return None, None, None


@st.cache_resource
def load_mediapipe():
    """Load MediaPipe face detection and mesh models"""
    try:
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # Use full range model (better for varying distances)
            min_detection_confidence=0.7
        )
        
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,  # Support up to 2 faces
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        return face_detection, face_mesh, mp_face_detection, mp_face_mesh
    except Exception as e:
        st.error(f"MediaPipe not available: {e}")
        return None, None, None, None


class EnhancedDrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        # Detection thresholds
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 15
        self.COUNTER = 0
        self.ALARM_ON = False
        self.MOBILE_COUNTER = 0
        self.MOBILE_THRESH = 10
        self.MOBILE_ALERT = False
        
        # Enhanced face tracking
        self.FACE_CONFIDENCE_THRESH = 0.7
        self.face_tracker = None
        self.face_lost_frames = 0
        self.MAX_FACE_LOST_FRAMES = 30
        
        # Load detectors
        self.face_cascade, self.eye_cascade, self.profile_cascade = load_detectors()
        self.face_detection, self.face_mesh, self.mp_face_detection, self.mp_face_mesh = load_mediapipe()
        
        # Smoothing for EAR values
        self.ear_history = deque(maxlen=5)
        
        # Metrics for dashboard
        self.current_ear = 0.3
        self.eyes_detected = 0
        self.faces_detected = 0
        self.frame_count = 0
        self.face_confidence = 0.0
        self.detection_method = "None"
        
        # Face region history for tracking stability
        self.last_face_region = None
        self.face_region_history = deque(maxlen=3)

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for face tracking"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0

    def smooth_face_region(self, current_region):
        """Smooth face region using history to reduce jitter"""
        if self.last_face_region is not None:
            # Calculate IoU with last detection
            iou = self.calculate_iou(current_region, self.last_face_region)
            
            # If IoU is high, use weighted average for smoothing
            if iou > 0.5:
                alpha = 0.7  # Smoothing factor
                x, y, w, h = current_region
                lx, ly, lw, lh = self.last_face_region
                
                smoothed_x = int(alpha * x + (1 - alpha) * lx)
                smoothed_y = int(alpha * y + (1 - alpha) * ly)
                smoothed_w = int(alpha * w + (1 - alpha) * lw)
                smoothed_h = int(alpha * h + (1 - alpha) * lh)
                
                current_region = (smoothed_x, smoothed_y, smoothed_w, smoothed_h)
        
        self.last_face_region = current_region
        self.face_region_history.append(current_region)
        return current_region

    def detect_face_mediapipe(self, frame):
        """Enhanced face detection using MediaPipe"""
        if self.face_detection is None:
            return [], 0.0, "MediaPipe unavailable"
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            faces = []
            max_confidence = 0.0
            
            if results.detections:
                for detection in results.detections:
                    confidence = detection.score[0]
                    if confidence > self.FACE_CONFIDENCE_THRESH:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Ensure coordinates are within frame bounds
                        x = max(0, x)
                        y = max(0, y)
                        width = min(width, w - x)
                        height = min(height, h - y)
                        
                        if width > 50 and height > 50:  # Minimum size filter
                            faces.append((x, y, width, height))
                            max_confidence = max(max_confidence, confidence)
            
            return faces, max_confidence, "MediaPipe"
        
        except Exception as e:
            return [], 0.0, f"MediaPipe error: {str(e)}"

    def detect_face_opencv_enhanced(self, frame):
        """Enhanced OpenCV face detection with multiple cascades"""
        if self.face_cascade is None:
            return [], 0.0, "OpenCV unavailable"
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        faces = []
        
        # Primary frontal face detection with enhanced parameters
        frontal_faces = self.face_cascade.detectMultiScale(
            enhanced_gray,
            scaleFactor=1.05,  # Smaller scale factor for better precision
            minNeighbors=6,    # Higher neighbors for better precision
            minSize=(80, 80),  # Reasonable minimum size
            maxSize=(400, 400), # Maximum size limit
            flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
        )
        
        # Add frontal faces
        for (x, y, w, h) in frontal_faces:
            faces.append((x, y, w, h))
        
        # If no frontal faces found, try profile detection
        if len(faces) == 0 and self.profile_cascade is not None:
            profile_faces = self.profile_cascade.detectMultiScale(
                enhanced_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
                maxSize=(300, 300)
            )
            
            for (x, y, w, h) in profile_faces:
                faces.append((x, y, w, h))
        
        # Filter overlapping detections
        if len(faces) > 1:
            faces = self.filter_overlapping_faces(faces)
        
        confidence = 0.8 if len(faces) > 0 else 0.0
        return faces, confidence, "OpenCV Enhanced"

    def filter_overlapping_faces(self, faces):
        """Filter out overlapping face detections"""
        if len(faces) <= 1:
            return faces
        
        # Sort by area (largest first)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        
        filtered_faces = []
        for face in faces:
            overlaps = False
            for existing_face in filtered_faces:
                if self.calculate_iou(face, existing_face) > 0.3:
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_faces.append(face)
        
        return filtered_faces

    def extract_eye_landmarks_mediapipe(self, frame, face_region, face_bbox):
        """Extract precise eye landmarks using MediaPipe Face Mesh"""
        if self.face_mesh is None:
            return [], []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            ear_values = []
            eye_regions = []
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get image dimensions
                    h, w, _ = frame.shape
                    
                    # Eye landmark indices (MediaPipe face mesh)
                    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    
                    # Extract left eye landmarks
                    left_eye_points = []
                    for idx in LEFT_EYE_INDICES:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        left_eye_points.append((x, y))
                    
                    # Extract right eye landmarks
                    right_eye_points = []
                    for idx in RIGHT_EYE_INDICES:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        right_eye_points.append((x, y))
                    
                    # Calculate EAR for both eyes
                    if len(left_eye_points) >= 6:
                        left_ear = self.calculate_ear_from_landmarks(left_eye_points)
                        ear_values.append(left_ear)
                        eye_regions.append(left_eye_points)
                    
                    if len(right_eye_points) >= 6:
                        right_ear = self.calculate_ear_from_landmarks(right_eye_points)
                        ear_values.append(right_ear)
                        eye_regions.append(right_eye_points)
            
            return ear_values, eye_regions
        
        except Exception as e:
            return [], []

    def calculate_ear_from_landmarks(self, eye_points):
        """Calculate EAR from eye landmark points"""
        if len(eye_points) < 6:
            return 0.3
        
        try:
            # Convert points to numpy array
            points = np.array(eye_points, dtype=np.float32)
            
            # Calculate vertical distances (height)
            vertical_dist_1 = np.linalg.norm(points[1] - points[5])
            vertical_dist_2 = np.linalg.norm(points[2] - points[4])
            
            # Calculate horizontal distance (width)
            horizontal_dist = np.linalg.norm(points[0] - points[3])
            
            # Calculate EAR
            if horizontal_dist > 0:
                ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
                return max(0.05, min(0.5, ear))
            
            return 0.3
        except:
            return 0.3

    def eye_aspect_ratio_opencv(self, eye_region):
        """Fallback EAR calculation using OpenCV methods"""
        if eye_region.size == 0:
            return 0.3
            
        # Convert to grayscale if needed
        if len(eye_region.shape) == 3:
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_eye = eye_region
        
        # Apply Gaussian blur and morphological operations
        blurred = cv2.GaussianBlur(gray_eye, (3, 3), 0)
        
        # Use adaptive thresholding for better results
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological closing to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Calculate ratio of white pixels to total pixels
        white_pixels = cv2.countNonZero(closed)
        total_pixels = closed.shape[0] * closed.shape[1]
        
        if total_pixels > 0:
            open_ratio = white_pixels / total_pixels
            # Scale to EAR-like values with better calibration
            ear = open_ratio * 0.4  # Adjusted scaling
            return max(0.05, min(0.5, ear))
        
        return 0.3

    def detect_eyes_opencv_enhanced(self, face_region, frame, face_x, face_y):
        """Enhanced eye detection using OpenCV"""
        if self.eye_cascade is None:
            return [], []
            
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_face = clahe.apply(gray_face)
        
        # Detect eyes with enhanced parameters
        eyes = self.eye_cascade.detectMultiScale(
            enhanced_face,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(15, 15),
            maxSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter eyes to get the best pair
        if len(eyes) > 2:
            # Sort by area and take the two largest
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        
        ear_values = []
        
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            # Extract eye region with padding
            padding = 5
            ey_start = max(0, ey - padding)
            ey_end = min(face_region.shape[0], ey + eh + padding)
            ex_start = max(0, ex - padding)
            ex_end = min(face_region.shape[1], ex + ew + padding)
            
            eye_region = face_region[ey_start:ey_end, ex_start:ex_end]
            
            if eye_region.size > 0:
                # Calculate EAR
                ear = self.eye_aspect_ratio_opencv(eye_region)
                ear_values.append(ear)
                
                # Draw eye rectangle with confidence-based color
                color = (0, 255, 0) if ear > self.EYE_AR_THRESH else (0, 165, 255)
                cv2.rectangle(frame, (face_x + ex, face_y + ey),
                            (face_x + ex + ew, face_y + ey + eh), color, 2)
                
                # Display EAR value
                cv2.putText(frame, f"EAR: {ear:.2f}", (face_x + ex, face_y + ey - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return ear_values, eyes

    def smooth_ear_values(self, ear_values):
        """Apply smoothing to EAR values"""
        if not ear_values:
            return 0.3
        
        avg_ear = sum(ear_values) / len(ear_values)
        self.ear_history.append(avg_ear)
        
        # Return smoothed average
        if len(self.ear_history) > 0:
            return sum(self.ear_history) / len(self.ear_history)
        return avg_ear

    def detect_mobile_phone_enhanced(self, frame, face_x, face_y, face_w, face_h):
        """Enhanced mobile phone detection with better algorithms"""
        mobile_detected = False
        
        # Define more precise ear regions based on face landmarks
        ear_width = int(face_w * 0.25)
        ear_height = int(face_h * 0.5)
        
        # Adjust ear regions based on face orientation
        right_ear_x = face_x + int(face_w * 0.9)
        right_ear_y = face_y + int(face_h * 0.25)
        
        left_ear_x = max(0, face_x - ear_width)
        left_ear_y = face_y + int(face_h * 0.25)
        
        # Convert to multiple color spaces for better detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Enhanced dark object detection
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 50])
        mask_hsv = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Additional mask for dark objects in LAB space
        lower_lab = np.array([0, 0, 0])
        upper_lab = np.array([120, 255, 120])
        mask_lab = cv2.inRange(lab, lower_lab, upper_lab)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_hsv, mask_lab)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        detection_confidence = 0
        
        # Check right ear region
        if right_ear_x + ear_width < frame.shape[1] and right_ear_y + ear_height < frame.shape[0]:
            right_roi = combined_mask[right_ear_y:right_ear_y + ear_height, 
                                   right_ear_x:right_ear_x + ear_width]
            if right_roi.size > 0:
                right_dark_pixels = cv2.countNonZero(right_roi)
                right_total_pixels = right_roi.shape[0] * right_roi.shape[1]
                right_ratio = right_dark_pixels / right_total_pixels if right_total_pixels > 0 else 0
                
                if right_ratio > 0.3:  # 30% of region is dark
                    detection_confidence += right_ratio
                    mobile_detected = True
                    cv2.rectangle(frame, (right_ear_x, right_ear_y),
                                (right_ear_x + ear_width, right_ear_y + ear_height), (0, 0, 255), 2)
                    cv2.putText(frame, f"PHONE? {right_ratio:.1%}", (right_ear_x, right_ear_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Check left ear region
        if left_ear_x >= 0 and left_ear_y + ear_height < frame.shape[0] and left_ear_x + ear_width < frame.shape[1]:
            left_roi = combined_mask[left_ear_y:left_ear_y + ear_height, 
                                  left_ear_x:left_ear_x + ear_width]
            if left_roi.size > 0:
                left_dark_pixels = cv2.countNonZero(left_roi)
                left_total_pixels = left_roi.shape[0] * left_roi.shape[1]
                left_ratio = left_dark_pixels / left_total_pixels if left_total_pixels > 0 else 0
                
                if left_ratio > 0.3:  # 30% of region is dark
                    detection_confidence += left_ratio
                    mobile_detected = True
                    cv2.rectangle(frame, (left_ear_x, left_ear_y),
                                (left_ear_x + ear_width, left_ear_y + ear_height), (0, 0, 255), 2)
                    cv2.putText(frame, f"PHONE? {left_ratio:.1%}", (left_ear_x, left_ear_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return mobile_detected, detection_confidence

    def recv(self, frame):
        """Main processing function with enhanced precision"""
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Try MediaPipe first for higher precision
        faces, confidence, method = self.detect_face_mediapipe(img)
        
        # Fallback to enhanced OpenCV if MediaPipe fails
        if len(faces) == 0:
            faces, confidence, method = self.detect_face_opencv_enhanced(img)
        
        self.faces_detected = len(faces)
        self.face_confidence = confidence
        self.detection_method = method
        
        ear_avg = 0.3
        mobile_detected = False
        mobile_confidence = 0
        
        if len(faces) > 0:
            # Process the most confident face
            x, y, w, h = faces[0]  # Take the first (most confident) face
            
            # Apply face region smoothing
            smoothed_face = self.smooth_face_region((x, y, w, h))
            x, y, w, h = smoothed_face
            
            # Reset face lost counter
            self.face_lost_frames = 0
            
            # Draw enhanced face rectangle with confidence
            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, f"Face ({confidence:.1%})", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Extract face region
            face_region = img[y:y + h, x:x + w]
            
            # Try MediaPipe eye detection first
            ear_values, eye_regions = self.extract_eye_landmarks_mediapipe(img, face_region, (x, y, w, h))
            
            # If MediaPipe eye detection fails, use enhanced OpenCV
            if len(ear_values) == 0:
                ear_values, eyes = self.detect_eyes_opencv_enhanced(face_region, img, x, y)
            else:
                # Draw MediaPipe eye landmarks
                for eye_points in eye_regions:
                    if len(eye_points) > 0:
                        pts = np.array(eye_points, np.int32)
                        cv2.polylines(img, [pts], True, (255, 255, 0), 1)
            
            self.eyes_detected = len(ear_values)
            
            # Apply EAR smoothing
            if len(ear_values) > 0:
                ear_avg = self.smooth_ear_values(ear_values)
            
            self.current_ear = ear_avg
            
            # Enhanced drowsiness detection
            if ear_avg < self.EYE_AR_THRESH:
                self.COUNTER += 1
                
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    if not self.ALARM_ON:
                        self.ALARM_ON = True
                    
                    # Draw enhanced drowsiness alert
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                    cv2.putText(img, "WAKE UP!", (10, 80),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    # Enhanced flashing effect
                    if self.frame_count % 8 < 4:  # Faster flashing
                        cv2.rectangle(img, (5, 5), (img.shape[1] - 5, img.shape[0] - 5), (0, 0, 255), 15)
            else:
                self.COUNTER = max(0, self.COUNTER - 1)  # Gradual decrease
                if self.COUNTER == 0:
                    self.ALARM_ON = False
            
            # Enhanced mobile phone detection
            mobile_detected, mobile_confidence = self.detect_mobile_phone_enhanced(img, x, y, w, h)
            
            if mobile_detected:
                self.MOBILE_COUNTER += 1
                if self.MOBILE_COUNTER >= self.MOBILE_THRESH:
                    if not self.MOBILE_ALERT:
                        self.MOBILE_ALERT = True
                    
                    cv2.putText(img, f"MOBILE PHONE DETECTED! ({mobile_confidence:.1%})", (10, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                    cv2.putText(img, "Please stop the car safely", (10, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            else:
                self.MOBILE_COUNTER = max(0, self.MOBILE_COUNTER - 1)
                if self.MOBILE_COUNTER == 0:
                    self.MOBILE_ALERT = False
        else:
            # No face detected - increment lost frames counter
            self.face_lost_frames += 1
            if self.face_lost_frames > self.MAX_FACE_LOST_FRAMES:
                # Reset tracking after too many lost frames
                self.last_face_region = None
                self.face_region_history.clear()
        
        # Enhanced on-screen metrics
        metrics_y_start = img.shape[0] - 120
        cv2.putText(img, f"Detection: {method}", (img.shape[1] - 250, metrics_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Face Conf: {confidence:.1%}", (img.shape[1] - 250, metrics_y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"EAR: {ear_avg:.3f}", (img.shape[1] - 250, metrics_y_start + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Eyes: {len(ear_values) if 'ear_values' in locals() else 0}", 
                   (img.shape[1] - 250, metrics_y_start + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Threshold: {self.EYE_AR_THRESH:.2f}", (img.shape[1] - 250, metrics_y_start + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Frame: {self.frame_count}", (img.shape[1] - 250, metrics_y_start + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    # Page configuration
    st.set_page_config(
        page_title="🚨layout="wide",
        initial_sidebar_state="expanded"
    )

    main()
