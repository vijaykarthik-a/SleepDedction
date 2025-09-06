import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import threading
import queue
import time
from collections import deque
from PIL import Image
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import mediapipe with better error handling
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe loaded successfully")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    logger.warning(f"MediaPipe import failed: {e}")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    logger.error(f"MediaPipe initialization failed: {e}")

# Load OpenCV cascades
@st.cache_resource
def load_detectors():
    """Load OpenCV face and eye detectors with optimized parameters"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Verify cascades are loaded properly
        if face_cascade.empty() or eye_cascade.empty() or profile_cascade.empty():
            st.error("OpenCV cascade classifiers failed to load")
            return None, None, None
            
        return face_cascade, eye_cascade, profile_cascade
    except Exception as e:
        st.error(f"Error loading OpenCV cascades: {e}")
        return None, None, None

@st.cache_resource
def load_mediapipe():
    """Load MediaPipe face detection, mesh, and hand models"""
    if not MEDIAPIPE_AVAILABLE:
        return None, None, None, None, None, None
        
    try:
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        mp_hands = mp.solutions.hands
        
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        return face_detection, face_mesh, hands, mp_face_detection, mp_face_mesh, mp_hands
    except Exception as e:
        st.error(f"MediaPipe initialization failed: {e}")
        return None, None, None, None, None, None

class EnhancedDetectionSystem(VideoProcessorBase):
    def __init__(self):
        # Eye detection thresholds
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 15
        self.EYE_COUNTER = 0
        self.DROWSY_ALARM = False
        
        # Phone detection thresholds
        self.PHONE_CONSEC_FRAMES = 10
        self.PHONE_COUNTER = 0
        self.PHONE_ALARM = False
        
        # Load detectors
        self.face_cascade, self.eye_cascade, self.profile_cascade = load_detectors()
        detection_result = load_mediapipe()
        if detection_result and detection_result[0] is not None:
            self.face_detection, self.face_mesh, self.hands, self.mp_face_detection, self.mp_face_mesh, self.mp_hands = detection_result
        else:
            self.face_detection = self.face_mesh = self.hands = None
            self.mp_face_detection = self.mp_face_mesh = self.mp_hands = None
        
        # Metrics
        self.current_ear = 0.3
        self.eyes_detected = 0
        self.faces_detected = 0
        self.hands_detected = 0
        self.frame_count = 0
        self.face_confidence = 0.0
        self.detection_method = "None"
        self.phone_detected = False
        self.head_pose = {"pitch": 0, "yaw": 0, "roll": 0}
        
        # History for stability
        self.ear_history = deque(maxlen=10)
        self.phone_history = deque(maxlen=5)
        
        # Processing state
        self.processing_active = True

    def calculate_ear_mediapipe(self, landmarks):
        """Calculate Eye Aspect Ratio using MediaPipe landmarks"""
        try:
            if not hasattr(landmarks, 'landmark') or len(landmarks.landmark) < 468:
                return 0.3
            
            # Left and right eye landmarks for MediaPipe face mesh
            LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            left_eye_points = []
            right_eye_points = []
            
            for idx in LEFT_EYE[:6]:  # Use first 6 points for simplicity
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    left_eye_points.append([point.x, point.y])
            
            for idx in RIGHT_EYE[:6]:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    right_eye_points.append([point.x, point.y])
            
            if len(left_eye_points) >= 6 and len(right_eye_points) >= 6:
                left_ear = self.calculate_single_ear(left_eye_points)
                right_ear = self.calculate_single_ear(right_eye_points)
                ear = (left_ear + right_ear) / 2.0
                return max(0.05, min(0.6, ear))
            
            return 0.3
            
        except Exception as e:
            logger.error(f"EAR calculation error: {e}")
            return 0.3

    def calculate_single_ear(self, eye_points):
        """Calculate EAR for a single eye"""
        try:
            if len(eye_points) < 6:
                return 0.3
            
            points = np.array(eye_points, dtype=np.float32)
            
            # Calculate vertical and horizontal distances
            vertical_1 = np.linalg.norm(points[1] - points[5])
            vertical_2 = np.linalg.norm(points[2] - points[4])
            horizontal = np.linalg.norm(points[0] - points[3])
            
            if horizontal > 1e-6:
                ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
                return max(0.05, min(0.6, ear))
            
            return 0.3
            
        except Exception as e:
            logger.error(f"Single EAR calculation error: {e}")
            return 0.3

    def detect_phone_usage(self, frame, hands_results):
        """Detect phone usage using hand position analysis"""
        try:
            if frame is None or hands_results is None:
                return False, 0.0, "No input"
                
            if not hasattr(hands_results, 'multi_hand_landmarks') or not hands_results.multi_hand_landmarks:
                return False, 0.0, "No hands detected"
            
            frame_height, frame_width = frame.shape[:2]
            phone_indicators = 0
            
            for hand_landmarks in hands_results.multi_hand_landmarks:
                if len(hand_landmarks.landmark) < 21:
                    continue
                
                # Get key hand points
                wrist = hand_landmarks.landmark[0]
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                
                # Convert to pixel coordinates
                wrist_x = int(np.clip(wrist.x * frame_width, 0, frame_width - 1))
                wrist_y = int(np.clip(wrist.y * frame_height, 0, frame_height - 1))
                thumb_x = int(np.clip(thumb_tip.x * frame_width, 0, frame_width - 1))
                thumb_y = int(np.clip(thumb_tip.y * frame_height, 0, frame_height - 1))
                index_x = int(np.clip(index_tip.x * frame_width, 0, frame_width - 1))
                index_y = int(np.clip(index_tip.y * frame_height, 0, frame_height - 1))
                
                # Phone holding indicators
                if 0.2 < wrist.y < 0.7:  # Hand in face/chest area
                    phone_indicators += 1
                
                if thumb_tip.y < wrist.y - 0.05 and index_tip.y < middle_tip.y:
                    phone_indicators += 1
                
                # Draw hand landmarks
                cv2.circle(frame, (wrist_x, wrist_y), 8, (255, 0, 255), -1)
                cv2.circle(frame, (thumb_x, thumb_y), 6, (0, 255, 255), -1)
                cv2.circle(frame, (index_x, index_y), 6, (0, 255, 255), -1)
                
                # Draw connecting lines
                cv2.line(frame, (wrist_x, wrist_y), (thumb_x, thumb_y), (255, 255, 0), 2)
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 255, 0), 2)
            
            phone_confidence = min(phone_indicators / 3.0, 1.0)
            phone_detected = phone_confidence > 0.6
            
            return phone_detected, phone_confidence, f"Indicators: {phone_indicators}/3"
            
        except Exception as e:
            logger.error(f"Phone detection error: {e}")
            return False, 0.0, f"Error: {str(e)}"

    def detect_face_mediapipe(self, frame):
        """Face detection using MediaPipe"""
        if not MEDIAPIPE_AVAILABLE or self.face_detection is None:
            return [], 0.0, "MediaPipe unavailable"
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            faces = []
            max_confidence = 0.0
            
            if results and hasattr(results, 'detections') and results.detections:
                for detection in results.detections:
                    if len(detection.score) == 0:
                        continue
                        
                    confidence = float(detection.score[0])
                    max_confidence = max(max_confidence, confidence)
                    
                    bbox = detection.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    
                    x = int(np.clip(bbox.xmin * w, 0, w - 1))
                    y = int(np.clip(bbox.ymin * h, 0, h - 1))
                    width = int(np.clip(bbox.width * w, 1, w - x))
                    height = int(np.clip(bbox.height * h, 1, h - y))
                    
                    if width > 0 and height > 0:
                        faces.append((x, y, width, height))
            
            return faces, max_confidence, "MediaPipe"
            
        except Exception as e:
            logger.error(f"MediaPipe face detection error: {e}")
            return [], 0.0, f"MediaPipe Error"

    def detect_face_opencv(self, frame):
        """Face detection using OpenCV"""
        if self.face_cascade is None:
            return [], 0.0, "OpenCV unavailable"
        
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            # Apply enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            faces = self.face_cascade.detectMultiScale(
                enhanced_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                maxSize=(400, 400)
            )
            
            confidence = 0.8 if len(faces) > 0 else 0.0
            return faces.tolist() if len(faces) > 0 else [], confidence, "OpenCV"
            
        except Exception as e:
            logger.error(f"OpenCV face detection error: {e}")
            return [], 0.0, f"OpenCV Error"

    def calculate_ear_opencv(self, eye_region):
        """Calculate EAR using OpenCV"""
        try:
            if eye_region is None or eye_region.size == 0:
                return 0.3
                
            if len(eye_region.shape) == 3:
                gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_eye = eye_region.copy()
            
            # Simple thresholding approach
            _, thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                if w > 0:
                    aspect_ratio = h / w
                    return max(0.05, min(0.6, aspect_ratio * 0.5))
            
            return 0.3
            
        except Exception as e:
            logger.error(f"OpenCV EAR calculation error: {e}")
            return 0.3

    def detect_eyes(self, face_region, frame, face_x, face_y):
        """Detect eyes in face region"""
        if self.eye_cascade is None:
            return []
        
        try:
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_region.copy()
            
            eyes = self.eye_cascade.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(15, 15),
                maxSize=(60, 60)
            )
            
            ear_values = []
            
            for (ex, ey, ew, eh) in eyes:
                # Extract eye region
                eye_region = face_region[ey:ey + eh, ex:ex + ew]
                
                if eye_region.size > 0:
                    ear = self.calculate_ear_opencv(eye_region)
                    ear_values.append(ear)
                    
                    # Draw eye rectangle
                    color = (0, 255, 0) if ear > self.EYE_AR_THRESH else (0, 0, 255)
                    abs_x1 = face_x + ex
                    abs_y1 = face_y + ey
                    abs_x2 = face_x + ex + ew
                    abs_y2 = face_y + ey + eh
                    
                    cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), color, 2)
                    cv2.putText(frame, f"{ear:.2f}", (abs_x1, abs_y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            return ear_values
            
        except Exception as e:
            logger.error(f"Eye detection error: {e}")
            return []

    def recv(self, frame):
        """Main processing function"""
        try:
            if not self.processing_active:
                return frame
                
            img = frame.to_ndarray(format="bgr24")
            
            if img is None or img.size == 0:
                return frame
            
            self.frame_count += 1
            
            # Face detection - try MediaPipe first, fallback to OpenCV
            faces = []
            confidence = 0.0
            method = "None"
            
            if MEDIAPIPE_AVAILABLE and self.face_detection is not None:
                faces, confidence, method = self.detect_face_mediapipe(img)
                
            if len(faces) == 0 and self.face_cascade is not None:
                faces, confidence, method = self.detect_face_opencv(img)
            
            self.faces_detected = len(faces)
            self.face_confidence = confidence
            self.detection_method = method
            
            # Hand detection for phone usage
            phone_detected = False
            phone_confidence = 0.0
            self.hands_detected = 0
            
            if MEDIAPIPE_AVAILABLE and self.hands is not None:
                try:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    hands_results = self.hands.process(rgb_img)
                    
                    if hands_results and hands_results.multi_hand_landmarks:
                        self.hands_detected = len(hands_results.multi_hand_landmarks)
                        phone_detected, phone_confidence, _ = self.detect_phone_usage(img, hands_results)
                        
                except Exception as e:
                    logger.error(f"Hand detection error: {e}")
            
            self.phone_detected = phone_detected
            
            # Eye analysis
            ear_avg = 0.3
            
            if len(faces) > 0:
                # Process first face
                face_data = faces[0]
                x, y, w, h = face_data[:4]
                
                # Validate coordinates
                x = max(0, min(img.shape[1] - 1, int(x)))
                y = max(0, min(img.shape[0] - 1, int(y)))
                w = max(1, min(img.shape[1] - x, int(w)))
                h = max(1, min(img.shape[0] - y, int(h)))
                
                x2 = min(img.shape[1], x + w)
                y2 = min(img.shape[0], y + h)
                
                # Draw face rectangle
                face_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.rectangle(img, (x, y), (x2, y2), face_color, 3)
                cv2.putText(img, f"Face ({confidence:.2f})", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
                
                # Extract face region and detect eyes
                if y2 > y and x2 > x:
                    face_region = img[y:y2, x:x2]
                    
                    if face_region.size > 0:
                        ear_values = self.detect_eyes(face_region, img, x, y)
                        self.eyes_detected = len(ear_values)
                        
                        if len(ear_values) > 0:
                            ear_avg = sum(ear_values) / len(ear_values)
                            self.ear_history.append(ear_avg)
                            
                            # Apply smoothing
                            if len(self.ear_history) > 3:
                                recent_ears = list(self.ear_history)[-5:]
                                ear_avg = sum(recent_ears) / len(recent_ears)
                
                self.current_ear = ear_avg
                
                # Drowsiness detection
                if ear_avg < self.EYE_AR_THRESH:
                    self.EYE_COUNTER += 1
                    
                    if self.EYE_COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        self.DROWSY_ALARM = True
                        
                        # Alert visualization
                        cv2.putText(img, "DROWSINESS ALERT!", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        cv2.putText(img, "WAKE UP!", (10, 110),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        
                        # Flashing border
                        if self.frame_count % 6 < 3:
                            cv2.rectangle(img, (5, 5), (img.shape[1]-5, img.shape[0]-5), (0, 0, 255), 8)
                else:
                    self.EYE_COUNTER = max(0, self.EYE_COUNTER - 2)
                    if self.EYE_COUNTER == 0:
                        self.DROWSY_ALARM = False
            
            # Phone detection alerts
            self.phone_history.append(phone_detected)
            consistent_phone = sum(self.phone_history) >= len(self.phone_history) * 0.6
            
            if consistent_phone:
                self.PHONE_COUNTER += 1
                if self.PHONE_COUNTER >= self.PHONE_CONSEC_FRAMES:
                    self.PHONE_ALARM = True
            else:
                self.PHONE_COUNTER = max(0, self.PHONE_COUNTER - 1)
                if self.PHONE_COUNTER == 0:
                    self.PHONE_ALARM = False
            
            if self.PHONE_ALARM:
                cv2.putText(img, "PHONE DETECTED!", (10, 200),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                cv2.putText(img, "PUT PHONE DOWN!", (10, 240),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                if self.frame_count % 8 < 4:
                    cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255, 0, 0), 6)
            
            # Display metrics
            metrics_y = img.shape[0] - 120
            ear_status = 'CLOSED' if ear_avg < self.EYE_AR_THRESH else 'OPEN'
            
            cv2.putText(img, f"EAR: {ear_avg:.3f} ({ear_status})", 
                       (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Eyes: {self.eyes_detected} | Faces: {self.faces_detected}", 
                       (10, metrics_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"Hands: {self.hands_detected} | Phone: {phone_confidence:.2f}", 
                       (10, metrics_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"Method: {method}", 
                       (10, metrics_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Status indicator
            status_x = img.shape[1] - 200
            if self.DROWSY_ALARM:
                cv2.putText(img, "DROWSY", (status_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif self.PHONE_ALARM:
                cv2.putText(img, "PHONE", (status_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(img, "ALERT", (status_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Main processing error: {e}")
            return frame


def process_uploaded_image(uploaded_file):
    """Process uploaded image for analysis"""
    try:
        if uploaded_file is None:
            return None, None
        
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Load detectors
        face_cascade, eye_cascade, _ = load_detectors()
        detection_result = load_mediapipe()
        
        results = {
            "faces": 0,
            "eyes": 0,
            "hands": 0,
            "phone_confidence": 0.0,
            "avg_ear": 0.3,
            "analysis": []
        }
        
        if face_cascade is not None:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(50, 50)
            )
            
            results["faces"] = len(faces)
            
            ear_values = []
            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img_array, "Face", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Detect eyes in face region
                face_region = img_array[y:y + h, x:x + w]
                
                if face_region.size > 0 and eye_cascade is not None:
                    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=4)
                    
                    results["eyes"] += len(eyes)
                    
                    for (ex, ey, ew, eh) in eyes:
                        eye_region = face_region[ey:ey + eh, ex:ex + ew]
                        
                        if eye_region.size > 0:
                            # Simple EAR estimation
                            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                            _, thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY)
                            
                            white_pixels = cv2.countNonZero(thresh)
                            total_pixels = thresh.shape[0] * thresh.shape[1]
                            
                            if total_pixels > 0:
                                ear = (white_pixels / total_pixels) * 0.5
                                ear_values.append(max(0.05, min(0.6, ear)))
                            
                            color = (0, 255, 0) if len(ear_values) > 0 and ear_values[-1] > 0.25 else (0, 0, 255)
                            cv2.rectangle(img_array, (x + ex, y + ey), (x + ex + ew, y + ey + eh), color, 2)
                            
                            if len(ear_values) > 0:
                                cv2.putText(img_array, f"EAR: {ear_values[-1]:.2f}", 
                                          (x + ex, y + ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if ear_values:
            results["avg_ear"] = sum(ear_values) / len(ear_values)
            results["analysis"].append(f"Average EAR: {results['avg_ear']:.3f}")
            results["analysis"].append("Eyes appear CLOSED" if results["avg_ear"] < 0.25 else "Eyes appear OPEN")
        
        # Hand/phone detection
        if MEDIAPIPE_AVAILABLE and detection_result and detection_result[2] is not None:
            hands_detector = detection_result[2]
            rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            hands_results = hands_detector.process(rgb_img)
            
            if hands_results and hands_results.multi_hand_landmarks:
                results["hands"] = len(hands_results.multi_hand_landmarks)
                
                phone_indicators = 0
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * img_array.shape[1])
                        y = int(landmark.y * img_array.shape[0])
                        cv2.circle(img_array, (x, y), 3, (255, 0, 255), -1)
                    
                    # Check for phone holding patterns
                    if len(hand_landmarks.landmark) >= 21:
                        wrist = hand_landmarks.landmark[0]
                        thumb_tip = hand_landmarks.landmark[4]
                        index_tip = hand_landmarks.landmark[8]
                        
                        if 0.2 < wrist.y < 0.7:  # Hand in face area
                            phone_indicators += 1
                        
                        if thumb_tip.y < wrist.y and index_tip.y < wrist.y:
                            phone_indicators += 1
                
                results["phone_confidence"] = min(phone_indicators / 2.0, 1.0)
                results["analysis"].append(f"Phone usage confidence: {results['phone_confidence']:.2f}")
                
                if results["phone_confidence"] > 0.5:
                    results["analysis"].append("⚠️ PHONE USAGE DETECTED")
        
        # Convert back to RGB for display
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        return img_array, results
        
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None, None


# Enhanced WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ],
    "iceCandidatePoolSize": 10
})


def main():
    """Main application with comprehensive driver monitoring"""
    st.set_page_config(
        page_title="Enhanced Driver Monitoring System",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .danger-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Enhanced Driver Monitoring System</h1>
        <p>AI-powered driver safety monitoring with drowsiness and phone usage detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Detection settings
    st.sidebar.subheader("Detection Settings")
    
    ear_threshold = st.sidebar.slider(
        "Eye Aspect Ratio Threshold",
        min_value=0.15,
        max_value=0.35,
        value=0.25,
        step=0.01,
        help="Lower values = more sensitive drowsiness detection"
    )
    
    drowsy_frames = st.sidebar.slider(
        "Drowsiness Detection Frames",
        min_value=5,
        max_value=30,
        value=15,
        help="Number of consecutive frames needed to trigger drowsiness alert"
    )
    
    phone_threshold = st.sidebar.slider(
        "Phone Detection Sensitivity",
        min_value=0.3,
        max_value=0.8,
        value=0.6,
        step=0.05,
        help="Confidence threshold for phone usage detection"
    )
    
    # System status
    st.sidebar.subheader("System Status")
    
    opencv_status = "✅ Available" if cv2 is not None else "❌ Not Available"
    mediapipe_status = "✅ Available" if MEDIAPIPE_AVAILABLE else "❌ Not Available"
    
    st.sidebar.markdown(f"""
    **OpenCV:** {opencv_status}  
    **MediaPipe:** {mediapipe_status}  
    **WebRTC:** ✅ Available
    """)
    
    # Camera troubleshooting
    st.sidebar.subheader("Camera Issues?")
    if st.sidebar.button("Troubleshooting Tips"):
        st.sidebar.info("""
        **If camera doesn't work:**
        1. Allow camera permissions
        2. Use Chrome browser
        3. Check HTTPS connection
        4. Refresh the page
        5. Try different camera settings
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Monitoring", "📸 Image Analysis", "🔧 Troubleshooting", "ℹ️ Information"])
    
    with tab1:
        st.header("📹 Real-Time Driver Monitoring")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Video Feed")
            
            # Camera test button
            if st.button("🔍 Test Camera Access"):
                st.info("Testing camera access... If this doesn't work, check the troubleshooting tab.")
            
            # WebRTC streamer with enhanced configuration
            try:
                webrtc_ctx = webrtc_streamer(
                    key="driver-monitoring-enhanced",
                    video_processor_factory=lambda: EnhancedDetectionSystem(),
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints={
                        "video": {
                            "width": {"min": 320, "ideal": 640, "max": 1280},
                            "height": {"min": 240, "ideal": 480, "max": 720},
                            "frameRate": {"min": 10, "ideal": 15, "max": 30},
                            "facingMode": "user"  # Front-facing camera
                        },
                        "audio": False
                    },
                    async_processing=True,
                    desired_playing_state=False,
                    video_html_attrs={
                        "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #007bff"},
                        "controls": False,
                        "autoplay": True,
                        "muted": True
                    }
                )
                
                # Display connection status
                if webrtc_ctx.state.playing:
                    st.success("✅ Camera connected successfully!")
                elif webrtc_ctx.state.signalling:
                    st.info("🔄 Connecting to camera...")
                else:
                    st.warning("⚠️ Camera not connected. Click 'START' to begin monitoring.")
                    
            except Exception as e:
                st.error(f"WebRTC initialization error: {e}")
                st.info("Try refreshing the page or check the troubleshooting tab.")
        
        with col2:
            st.subheader("📊 Live Metrics")
            
            # Placeholder containers for real-time updates
            ear_container = st.empty()
            detection_container = st.empty()
            alert_container = st.empty()
            status_container = st.empty()
            
            # Display default metrics when not running
            if 'webrtc_ctx' not in locals() or not webrtc_ctx.state.playing:
                with ear_container.container():
                    st.markdown("""
                    <div class="metric-card">
                        <h4>👁️ Eye Status: Waiting...</h4>
                        <p><strong>EAR Value:</strong> ---</p>
                        <p><strong>Threshold:</strong> """ + str(ear_threshold) + """</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with detection_container.container():
                    st.markdown("""
                    <div class="metric-card">
                        <h4>🔍 Detection Stats</h4>
                        <p><strong>Faces:</strong> 0 | <strong>Eyes:</strong> 0</p>
                        <p><strong>Hands:</strong> 0</p>
                        <p><strong>Method:</strong> Waiting...</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with alert_container.container():
                    st.markdown("""
                    <div class="warning-box">
                        <h3>📹 Ready to Monitor</h3>
                        <p>Click START to begin driver monitoring</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.header("📸 Static Image Analysis")
        
        # File uploader with better error handling
        uploaded_file = st.file_uploader(
            "Upload a driver image for analysis",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear photo showing the driver's face"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Analyzing image..."):
                    processed_img, analysis_results = process_uploaded_image(uploaded_file)
                
                if processed_img is not None and analysis_results is not None:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📊 Analysis Results")
                        st.image(processed_img, caption="Processed Image with Detections", use_column_width=True)
                    
                    with col2:
                        st.subheader("📈 Detection Metrics")
                        
                        # Display results with better formatting
                        faces = analysis_results.get("faces", 0)
                        eyes = analysis_results.get("eyes", 0)
                        hands = analysis_results.get("hands", 0)
                        phone_conf = analysis_results.get("phone_confidence", 0.0)
                        avg_ear = analysis_results.get("avg_ear", 0.3)
                        
                        # Metrics display
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Faces", faces)
                            st.metric("Eyes", eyes)
                            st.metric("Hands", hands)
                        with col_b:
                            st.metric("Avg EAR", f"{avg_ear:.3f}")
                            st.metric("Phone Confidence", f"{phone_conf:.2f}")
                        
                        # Status assessment
                        st.subheader("🔍 Analysis Summary")
                        
                        if avg_ear < 0.25:
                            st.markdown("""
                            <div class="danger-box">
                                <strong>🚨 DROWSINESS DETECTED</strong><br>
                                Eyes appear closed - driver may be drowsy
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="success-box">
                                <strong>✅ EYES OPEN</strong><br>
                                Driver appears alert
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if phone_conf > 0.5:
                            st.markdown("""
                            <div class="danger-box">
                                <strong>📱 PHONE USAGE DETECTED</strong><br>
                                Possible phone handling detected
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="success-box">
                                <strong>✅ NO PHONE DETECTED</strong><br>
                                No phone usage indicators
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed analysis
                        if analysis_results.get("analysis"):
                            st.subheader("🔍 Detailed Analysis")
                            for item in analysis_results["analysis"]:
                                st.text(f"• {item}")
                                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.info("Please try uploading a different image or check the file format.")
    
    with tab3:
        st.header("🔧 Troubleshooting Camera Issues")
        
        st.markdown("""
        ### Common Camera Problems and Solutions
        
        **Problem 1: "Camera not working" or black screen**
        """)
        
        with st.expander("🔍 Browser Permissions", expanded=True):
            st.markdown("""
            **Steps to fix:**
            1. Look for camera icon in your browser's address bar
            2. Click on it and select "Allow" for camera access
            3. Refresh this page after granting permissions
            4. If no camera icon appears, check browser settings
            
            **Chrome:** Settings → Privacy and Security → Site Settings → Camera
            **Firefox:** Settings → Privacy & Security → Permissions → Camera
            **Edge:** Settings → Site Permissions → Camera
            """)
        
        with st.expander("🌐 HTTPS and Browser Issues"):
            st.markdown("""
            **Requirements:**
            - Must use HTTPS (your Streamlit Cloud app should automatically have this)
            - Recommended browsers: Chrome, Firefox, Edge
            - Avoid Safari (limited WebRTC support)
            
            **If still not working:**
            1. Try opening in an incognito/private window
            2. Clear browser cache and cookies
            3. Disable browser extensions temporarily
            4. Check if camera works on other websites (like meet.google.com)
            """)
        
        with st.expander("💻 System and Hardware Issues"):
            st.markdown("""
            **Check these:**
            1. Camera is not being used by other applications (Zoom, Teams, etc.)
            2. Camera drivers are up to date
            3. Try unplugging and reconnecting external cameras
            4. For laptops, check if camera is physically blocked/disabled
            5. Restart your browser completely
            """)
        
        with st.expander("🔬 Technical Debugging"):
            st.markdown("""
            **For advanced users:**
            1. Open browser Developer Tools (F12)
            2. Go to Console tab
            3. Look for WebRTC or camera-related errors
            4. Check if `navigator.mediaDevices` is available
            5. Test with: `navigator.mediaDevices.getUserMedia({video: true})`
            """)
        
        st.markdown("### Quick Camera Test")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Test Basic Camera Access"):
                st.info("Click START on the main tab to test camera access.")
        
        with col2:
            if st.button("📋 Show System Info"):
                st.code(f"""
System Information:
- OpenCV: {'Available' if cv2 is not None else 'Not Available'}
- MediaPipe: {'Available' if MEDIAPIPE_AVAILABLE else 'Not Available'}
- Streamlit: {st.__version__}
                """)
        
        st.markdown("### Alternative Solutions")
        
        st.info("""
        **If camera still doesn't work:**
        1. Use the "Image Analysis" tab to upload photos instead
        2. Try using a different device or browser
        3. Check if your organization has camera usage restrictions
        4. Consider using a local development environment
        """)
    
    with tab4:
        st.header("ℹ️ System Information & Usage Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Detection Features")
            st.markdown("""
            **Drowsiness Detection:**
            - Eye Aspect Ratio (EAR) monitoring
            - Blink pattern analysis
            - Multiple detection algorithms
            - Real-time alerting system
            
            **Phone Usage Detection:**
            - Hand gesture recognition
            - Phone holding patterns
            - Grip analysis
            - Multi-indicator scoring
            
            **Face Detection:**
            - MediaPipe AI models
            - OpenCV Haar cascades
            - Multi-angle face detection
            - Confidence scoring
            """)
        
        with col2:
            st.subheader("⚡ Technical Specifications")
            st.markdown("""
            **Algorithms:**
            - MediaPipe Face Detection/Mesh
            - MediaPipe Hands
            - OpenCV Computer Vision
            - Advanced image processing
            
            **Performance:**
            - Real-time processing (15-30 FPS)
            - Adaptive thresholding
            - Error recovery systems
            - Multi-browser support
            
            **Supported Formats:**
            - Live webcam feed
            - JPG, PNG, BMP images
            - HD video processing
            """)
        
        st.subheader("📋 How to Use This System")
        
        with st.expander("🚗 For Driver Monitoring"):
            st.markdown("""
            **Setup:**
            1. Position camera at eye level
            2. Ensure good lighting
            3. Minimize background distractions
            4. Test camera access first
            
            **During Monitoring:**
            - Keep face visible to camera
            - Watch for drowsiness alerts
            - Monitor phone usage warnings
            - Adjust sensitivity settings as needed
            
            **Best Practices:**
            - Use in well-lit environments
            - Keep camera lens clean
            - Check system regularly
            - Take breaks as recommended
            """)
        
        with st.expander("📸 For Image Analysis"):
            st.markdown("""
            **Image Requirements:**
            - Clear, well-lit photos
            - Face clearly visible
            - Minimal motion blur
            - JPG, PNG, or BMP format
            
            **Analysis Features:**
            - Automatic face detection
            - Eye state analysis
            - Hand/phone detection
            - Detailed metrics report
            """)
        
        st.subheader("⚙️ System Requirements")
        
        requirements_info = {
            "Browser": "Chrome (recommended), Firefox, Edge",
            "Camera": "Any webcam with 640x480 or higher resolution",
            "Internet": "Stable connection for real-time processing",
            "Lighting": "Good ambient lighting for accurate detection",
            "Python Libraries": "OpenCV, MediaPipe, Streamlit-WebRTC"
        }
        
        for requirement, details in requirements_info.items():
            st.text(f"• {requirement}: {details}")
        
        st.subheader("⚠️ Important Notes")
        st.warning("""
        This system is designed for educational and research purposes. 
        For production use in vehicles, additional safety measures and 
        redundant systems should be implemented. Always prioritize 
        actual driver training and safe driving practices.
        """)


if __name__ == "__main__":
    main()
