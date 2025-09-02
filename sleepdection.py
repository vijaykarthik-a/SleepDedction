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

# Import mediapipe with better error handling
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded successfully")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️ MediaPipe import failed: {e}")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️ MediaPipe initialization failed: {e}")


# Load OpenCV cascades with enhanced parameters
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
        if detection_result[0] is not None:
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

    def calculate_ear_mediapipe(self, landmarks):
        """Calculate Eye Aspect Ratio using MediaPipe landmarks with improved accuracy"""
        try:
            # Validate landmarks input
            if not hasattr(landmarks, 'landmark') or len(landmarks.landmark) == 0:
                return 0.3
            
            # Left eye landmarks (indices for MediaPipe face mesh)
            LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Get coordinates with bounds checking
            left_eye_points = []
            right_eye_points = []
            
            landmark_count = len(landmarks.landmark)
            
            for idx in LEFT_EYE:
                if idx < landmark_count:
                    point = landmarks.landmark[idx]
                    left_eye_points.append([point.x, point.y])
            
            for idx in RIGHT_EYE:
                if idx < landmark_count:
                    point = landmarks.landmark[idx]
                    right_eye_points.append([point.x, point.y])
            
            if len(left_eye_points) < 6 or len(right_eye_points) < 6:
                return 0.3
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_single_ear(left_eye_points)
            right_ear = self.calculate_single_ear(right_eye_points)
            
            # Return average with bounds checking
            ear = (left_ear + right_ear) / 2.0
            return max(0.05, min(0.6, ear))
            
        except Exception as e:
            print(f"EAR calculation error: {e}")
            return 0.3

    def calculate_single_ear(self, eye_points):
        """Calculate EAR for a single eye with enhanced error handling"""
        try:
            if len(eye_points) < 6:
                return 0.3
            
            # Convert to numpy array with proper error handling
            points = np.array(eye_points, dtype=np.float32)
            
            # Validate array shape
            if points.shape[0] < 6 or points.shape[1] != 2:
                return 0.3
            
            # Calculate vertical distances (approximation)
            # Use safe indexing to prevent array errors
            try:
                vertical_1 = np.linalg.norm(points[1] - points[5])
                vertical_2 = np.linalg.norm(points[2] - points[4])
                horizontal = np.linalg.norm(points[0] - points[3])
            except IndexError:
                # Fallback to basic calculation with available points
                if len(points) >= 4:
                    vertical_1 = np.linalg.norm(points[1] - points[3])
                    vertical_2 = vertical_1  # Use same value
                    horizontal = np.linalg.norm(points[0] - points[2])
                else:
                    return 0.3
            
            # Prevent division by zero
            if horizontal == 0 or horizontal < 1e-6:
                return 0.3
            
            # EAR calculation with bounds checking
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            
            # Ensure reasonable EAR values
            if np.isnan(ear) or np.isinf(ear):
                return 0.3
                
            return max(0.05, min(0.6, ear))
            
        except Exception as e:
            print(f"Single EAR calculation error: {e}")
            return 0.3

    def detect_phone_usage(self, frame, hands_results):
        """Enhanced phone detection using hand position and posture analysis"""
        phone_indicators = 0
        phone_confidence = 0.0
        
        # Validate inputs
        if frame is None or frame.size == 0:
            return False, 0.0, "Invalid frame"
            
        if hands_results is None or not hasattr(hands_results, 'multi_hand_landmarks') or not hands_results.multi_hand_landmarks:
            return False, 0.0, "No hands detected"
        
        try:
            # Safely get frame dimensions
            if len(frame.shape) >= 2:
                frame_height, frame_width = frame.shape[:2]
            else:
                return False, 0.0, "Invalid frame shape"
            
            # Validate frame dimensions
            if frame_width <= 0 or frame_height <= 0:
                return False, 0.0, "Invalid frame dimensions"
            
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Validate hand landmarks
                if not hasattr(hand_landmarks, 'landmark') or len(hand_landmarks.landmark) < 21:
                    continue
                
                # Get key hand points with bounds checking
                try:
                    wrist = hand_landmarks.landmark[0]
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    middle_tip = hand_landmarks.landmark[12]
                    ring_tip = hand_landmarks.landmark[16]
                    pinky_tip = hand_landmarks.landmark[20]
                except IndexError:
                    continue
                
                # Convert to pixel coordinates with bounds checking
                wrist_x = int(np.clip(wrist.x * frame_width, 0, frame_width - 1))
                wrist_y = int(np.clip(wrist.y * frame_height, 0, frame_height - 1))
                thumb_x = int(np.clip(thumb_tip.x * frame_width, 0, frame_width - 1))
                thumb_y = int(np.clip(thumb_tip.y * frame_height, 0, frame_height - 1))
                index_x = int(np.clip(index_tip.x * frame_width, 0, frame_width - 1))
                index_y = int(np.clip(index_tip.y * frame_height, 0, frame_height - 1))
                
                wrist_pos = (wrist_x, wrist_y)
                thumb_pos = (thumb_x, thumb_y)
                index_pos = (index_x, index_y)
                
                # Phone holding indicators with bounds checking
                
                # 1. Hand position (near face level)
                if 0.2 < wrist.y < 0.7:  # Hand in face/chest area
                    phone_indicators += 1
                
                # 2. Finger configuration (thumb and index extended, others curled)
                thumb_extended = thumb_tip.y < wrist.y - 0.05
                index_extended = index_tip.y < middle_tip.y
                
                if thumb_extended and index_extended:
                    phone_indicators += 2
                
                # 3. Hand orientation (horizontal-ish position) with error handling
                try:
                    if thumb_pos[0] != index_pos[0] or thumb_pos[1] != index_pos[1]:
                        hand_angle = math.atan2(thumb_pos[1] - index_pos[1], thumb_pos[0] - index_pos[0])
                        hand_angle_deg = abs(math.degrees(hand_angle))
                        
                        if 30 < hand_angle_deg < 150:  # Horizontal-ish orientation
                            phone_indicators += 1
                except (ValueError, ZeroDivisionError):
                    pass
                
                # 4. Distance between thumb and fingers (gripping motion)
                try:
                    grip_dist = math.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
                    
                    if 50 < grip_dist < 200:  # Appropriate grip distance
                        phone_indicators += 1
                except (ValueError, OverflowError):
                    pass
                
                # 5. Hand stability (minimal movement)
                if abs(wrist.x - 0.5) < 0.3:  # Hand near center (stable position)
                    phone_indicators += 1
                
                # Draw hand landmarks with bounds checking
                if 0 <= wrist_pos[0] < frame_width and 0 <= wrist_pos[1] < frame_height:
                    cv2.circle(frame, wrist_pos, 8, (255, 0, 255), -1)
                if 0 <= thumb_pos[0] < frame_width and 0 <= thumb_pos[1] < frame_height:
                    cv2.circle(frame, thumb_pos, 6, (0, 255, 255), -1)
                if 0 <= index_pos[0] < frame_width and 0 <= index_pos[1] < frame_height:
                    cv2.circle(frame, index_pos, 6, (0, 255, 255), -1)
                
                # Draw connecting lines for grip visualization with bounds checking
                try:
                    cv2.line(frame, wrist_pos, thumb_pos, (255, 255, 0), 2)
                    cv2.line(frame, thumb_pos, index_pos, (255, 255, 0), 2)
                except cv2.error:
                    pass  # Skip drawing if coordinates are invalid
            
            # Calculate confidence based on indicators
            phone_confidence = min(phone_indicators / 5.0, 1.0)
            phone_detected = phone_confidence > 0.6
            
            return phone_detected, phone_confidence, f"Indicators: {phone_indicators}/5"
            
        except Exception as e:
            print(f"Phone detection error: {e}")
            return False, 0.0, f"Error: {str(e)}"

    def analyze_head_pose(self, landmarks, frame_width, frame_height):
        """Analyze head pose to detect phone usage patterns with error handling"""
        try:
            # Validate inputs
            if not hasattr(landmarks, 'landmark') or len(landmarks.landmark) < 468:
                return {"tilt": 0, "yaw_offset": 0, "looking_down": False, "head_turned": False}
            
            if frame_width <= 0 or frame_height <= 0:
                return {"tilt": 0, "yaw_offset": 0, "looking_down": False, "head_turned": False}
            
            # Key facial landmarks for head pose with bounds checking
            landmark_indices = {
                'nose_tip': 1,
                'chin': 175,
                'left_eye': 33,
                'right_eye': 263,
                'left_mouth': 61,
                'right_mouth': 291
            }
            
            # Extract landmarks safely
            landmark_points = {}
            for name, idx in landmark_indices.items():
                if idx < len(landmarks.landmark):
                    landmark_points[name] = landmarks.landmark[idx]
                else:
                    # Return default values if landmarks are incomplete
                    return {"tilt": 0, "yaw_offset": 0, "looking_down": False, "head_turned": False}
            
            # Calculate head tilt (looking down at phone)
            eye_center_y = (landmark_points['left_eye'].y + landmark_points['right_eye'].y) / 2
            mouth_center_y = (landmark_points['left_mouth'].y + landmark_points['right_mouth'].y) / 2
            
            # Head tilt angle (positive = looking down) with division by zero check
            nose_chin_diff = abs(landmark_points['nose_tip'].y - landmark_points['chin'].y)
            if nose_chin_diff > 1e-6:  # Prevent division by zero
                tilt_ratio = (mouth_center_y - eye_center_y) / nose_chin_diff
            else:
                tilt_ratio = 0
            
            # Side-to-side head movement
            face_center_x = (landmark_points['left_eye'].x + landmark_points['right_eye'].x) / 2
            yaw_offset = abs(face_center_x - 0.5)  # Deviation from center
            
            self.head_pose = {
                "tilt": tilt_ratio,
                "yaw_offset": yaw_offset,
                "looking_down": tilt_ratio > 0.15,  # Threshold for looking down
                "head_turned": yaw_offset > 0.2     # Threshold for head turned away
            }
            
            return self.head_pose
            
        except Exception as e:
            print(f"Head pose analysis error: {e}")
            return {"tilt": 0, "yaw_offset": 0, "looking_down": False, "head_turned": False}

    def calculate_ear_mediapipe(self, landmarks):
        """Calculate Eye Aspect Ratio using MediaPipe landmarks with improved accuracy"""
        try:
            # Validate landmarks input
            if not hasattr(landmarks, 'landmark') or len(landmarks.landmark) == 0:
                return 0.3
            
            # Left eye landmarks (indices for MediaPipe face mesh)
            LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Get coordinates with bounds checking
            left_eye_points = []
            right_eye_points = []
            
            landmark_count = len(landmarks.landmark)
            
            for idx in LEFT_EYE:
                if idx < landmark_count:
                    point = landmarks.landmark[idx]
                    left_eye_points.append([point.x, point.y])
            
            for idx in RIGHT_EYE:
                if idx < landmark_count:
                    point = landmarks.landmark[idx]
                    right_eye_points.append([point.x, point.y])
            
            if len(left_eye_points) < 6 or len(right_eye_points) < 6:
                return 0.3
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_single_ear(left_eye_points)
            right_ear = self.calculate_single_ear(right_eye_points)
            
            # Return average with bounds checking
            ear = (left_ear + right_ear) / 2.0
            return max(0.05, min(0.6, ear))
            
        except Exception as e:
            print(f"EAR calculation error: {e}")
            return 0.3

    def calculate_single_ear(self, eye_points):
        """Calculate EAR for a single eye with enhanced error handling"""
        try:
            if len(eye_points) < 6:
                return 0.3
            
            # Convert to numpy array with proper error handling
            points = np.array(eye_points, dtype=np.float32)
            
            # Validate array shape
            if points.shape[0] < 6 or points.shape[1] != 2:
                return 0.3
            
            # Calculate vertical distances (approximation)
            # Use safe indexing to prevent array errors
            try:
                vertical_1 = np.linalg.norm(points[1] - points[5])
                vertical_2 = np.linalg.norm(points[2] - points[4])
                horizontal = np.linalg.norm(points[0] - points[3])
            except IndexError:
                # Fallback to basic calculation with available points
                if len(points) >= 4:
                    vertical_1 = np.linalg.norm(points[1] - points[3])
                    vertical_2 = vertical_1  # Use same value
                    horizontal = np.linalg.norm(points[0] - points[2])
                else:
                    return 0.3
            
            # Prevent division by zero
            if horizontal == 0 or horizontal < 1e-6:
                return 0.3
            
            # EAR calculation with bounds checking
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            
            # Ensure reasonable EAR values
            if np.isnan(ear) or np.isinf(ear):
                return 0.3
                
            return max(0.05, min(0.6, ear))
            
        except Exception as e:
            print(f"Single EAR calculation error: {e}")
            return 0.3

    def detect_phone_usage(self, frame, hands_results):
        """Enhanced phone detection using hand position and posture analysis"""
        phone_indicators = 0
        phone_confidence = 0.0
        
        # Validate inputs
        if frame is None or frame.size == 0:
            return False, 0.0, "Invalid frame"
            
        if hands_results is None or not hasattr(hands_results, 'multi_hand_landmarks') or not hands_results.multi_hand_landmarks:
            return False, 0.0, "No hands detected"
        
        try:
            # Safely get frame dimensions
            if len(frame.shape) >= 2:
                frame_height, frame_width = frame.shape[:2]
            else:
                return False, 0.0, "Invalid frame shape"
            
            # Validate frame dimensions
            if frame_width <= 0 or frame_height <= 0:
                return False, 0.0, "Invalid frame dimensions"
            
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Validate hand landmarks
                if not hasattr(hand_landmarks, 'landmark') or len(hand_landmarks.landmark) < 21:
                    continue
                
                # Get key hand points with bounds checking
                try:
                    wrist = hand_landmarks.landmark[0]
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    middle_tip = hand_landmarks.landmark[12]
                    ring_tip = hand_landmarks.landmark[16]
                    pinky_tip = hand_landmarks.landmark[20]
                except IndexError:
                    continue
                
                # Convert to pixel coordinates with bounds checking
                wrist_x = int(np.clip(wrist.x * frame_width, 0, frame_width - 1))
                wrist_y = int(np.clip(wrist.y * frame_height, 0, frame_height - 1))
                thumb_x = int(np.clip(thumb_tip.x * frame_width, 0, frame_width - 1))
                thumb_y = int(np.clip(thumb_tip.y * frame_height, 0, frame_height - 1))
                index_x = int(np.clip(index_tip.x * frame_width, 0, frame_width - 1))
                index_y = int(np.clip(index_tip.y * frame_height, 0, frame_height - 1))
                
                wrist_pos = (wrist_x, wrist_y)
                thumb_pos = (thumb_x, thumb_y)
                index_pos = (index_x, index_y)
                
                # Phone holding indicators with bounds checking
                
                # 1. Hand position (near face level)
                if 0.2 < wrist.y < 0.7:  # Hand in face/chest area
                    phone_indicators += 1
                
                # 2. Finger configuration (thumb and index extended, others curled)
                thumb_extended = thumb_tip.y < wrist.y - 0.05
                index_extended = index_tip.y < middle_tip.y
                
                if thumb_extended and index_extended:
                    phone_indicators += 2
                
                # 3. Hand orientation (horizontal-ish position) with error handling
                try:
                    if thumb_pos[0] != index_pos[0] or thumb_pos[1] != index_pos[1]:
                        hand_angle = math.atan2(thumb_pos[1] - index_pos[1], thumb_pos[0] - index_pos[0])
                        hand_angle_deg = abs(math.degrees(hand_angle))
                        
                        if 30 < hand_angle_deg < 150:  # Horizontal-ish orientation
                            phone_indicators += 1
                except (ValueError, ZeroDivisionError):
                    pass
                
                # 4. Distance between thumb and fingers (gripping motion)
                try:
                    grip_dist = math.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
                    
                    if 50 < grip_dist < 200:  # Appropriate grip distance
                        phone_indicators += 1
                except (ValueError, OverflowError):
                    pass
                
                # 5. Hand stability (minimal movement)
                if abs(wrist.x - 0.5) < 0.3:  # Hand near center (stable position)
                    phone_indicators += 1
                
                # Draw hand landmarks with bounds checking
                if 0 <= wrist_pos[0] < frame_width and 0 <= wrist_pos[1] < frame_height:
                    cv2.circle(frame, wrist_pos, 8, (255, 0, 255), -1)
                if 0 <= thumb_pos[0] < frame_width and 0 <= thumb_pos[1] < frame_height:
                    cv2.circle(frame, thumb_pos, 6, (0, 255, 255), -1)
                if 0 <= index_pos[0] < frame_width and 0 <= index_pos[1] < frame_height:
                    cv2.circle(frame, index_pos, 6, (0, 255, 255), -1)
                
                # Draw connecting lines for grip visualization with bounds checking
                try:
                    cv2.line(frame, wrist_pos, thumb_pos, (255, 255, 0), 2)
                    cv2.line(frame, thumb_pos, index_pos, (255, 255, 0), 2)
                except cv2.error:
                    pass  # Skip drawing if coordinates are invalid
            
            # Calculate confidence based on indicators
            phone_confidence = min(phone_indicators / 5.0, 1.0)
            phone_detected = phone_confidence > 0.6
            
            return phone_detected, phone_confidence, f"Indicators: {phone_indicators}/5"
            
        except Exception as e:
            print(f"Phone detection error: {e}")
            return False, 0.0, f"Error: {str(e)}"

    def analyze_head_pose(self, landmarks, frame_width, frame_height):
        """Analyze head pose to detect phone usage patterns with error handling"""
        try:
            # Validate inputs
            if not hasattr(landmarks, 'landmark') or len(landmarks.landmark) < 468:
                return {"tilt": 0, "yaw_offset": 0, "looking_down": False, "head_turned": False}
            
            if frame_width <= 0 or frame_height <= 0:
                return {"tilt": 0, "yaw_offset": 0, "looking_down": False, "head_turned": False}
            
            # Key facial landmarks for head pose with bounds checking
            landmark_indices = {
                'nose_tip': 1,
                'chin': 175,
                'left_eye': 33,
                'right_eye': 263,
                'left_mouth': 61,
                'right_mouth': 291
            }
            
            # Extract landmarks safely
            landmark_points = {}
            for name, idx in landmark_indices.items():
                if idx < len(landmarks.landmark):
                    landmark_points[name] = landmarks.landmark[idx]
                else:
                    # Return default values if landmarks are incomplete
                    return {"tilt": 0, "yaw_offset": 0, "looking_down": False, "head_turned": False}
            
            # Calculate head tilt (looking down at phone)
            eye_center_y = (landmark_points['left_eye'].y + landmark_points['right_eye'].y) / 2
            mouth_center_y = (landmark_points['left_mouth'].y + landmark_points['right_mouth'].y) / 2
            
            # Head tilt angle (positive = looking down) with division by zero check
            nose_chin_diff = abs(landmark_points['nose_tip'].y - landmark_points['chin'].y)
            if nose_chin_diff > 1e-6:  # Prevent division by zero
                tilt_ratio = (mouth_center_y - eye_center_y) / nose_chin_diff
            else:
                tilt_ratio = 0
            
            # Side-to-side head movement
            face_center_x = (landmark_points['left_eye'].x + landmark_points['right_eye'].x) / 2
            yaw_offset = abs(face_center_x - 0.5)  # Deviation from center
            
            self.head_pose = {
                "tilt": tilt_ratio,
                "yaw_offset": yaw_offset,
                "looking_down": tilt_ratio > 0.15,  # Threshold for looking down
                "head_turned": yaw_offset > 0.2     # Threshold for head turned away
            }
            
            return self.head_pose
            
        except Exception as e:
            print(f"Head pose analysis error: {e}")
            return {"tilt": 0, "yaw_offset": 0, "looking_down": False, "head_turned": False}

    def detect_face_mediapipe(self, frame):
        """Enhanced face detection using MediaPipe with array error handling"""
        if not MEDIAPIPE_AVAILABLE or self.face_detection is None:
            return [], 0.0, "MediaPipe unavailable"
        
        # Validate input frame
        if frame is None or frame.size == 0:
            return [], 0.0, "Invalid frame"
        
        # Ensure frame has correct shape
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return [], 0.0, "Invalid frame format"
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            faces = []
            max_confidence = 0.0
            
            if results and hasattr(results, 'detections') and results.detections:
                for detection in results.detections:
                    # Validate detection object
                    if not hasattr(detection, 'score') or not hasattr(detection, 'location_data'):
                        continue
                    
                    if len(detection.score) == 0:
                        continue
                        
                    confidence = float(detection.score[0])
                    max_confidence = max(max_confidence, confidence)
                    
                    # Get bounding box with error handling
                    if not hasattr(detection.location_data, 'relative_bounding_box'):
                        continue
                    
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Validate frame dimensions
                    if len(frame.shape) < 2:
                        continue
                    
                    h, w = frame.shape[:2]
                    
                    # Calculate coordinates with bounds checking
                    x = int(np.clip(bbox.xmin * w, 0, w - 1))
                    y = int(np.clip(bbox.ymin * h, 0, h - 1))
                    width = int(np.clip(bbox.width * w, 1, w - x))
                    height = int(np.clip(bbox.height * h, 1, h - y))
                    
                    # Validate bounding box
                    if width > 0 and height > 0 and x + width <= w and y + height <= h:
                        faces.append((x, y, width, height))
            
            return faces, max_confidence, "MediaPipe"
            
        except Exception as e:
            print(f"MediaPipe face detection error: {e}")
            return [], 0.0, f"MediaPipe Error: {str(e)}"

    def detect_face_opencv(self, frame):
        """Enhanced OpenCV face detection with multiple cascades and error handling"""
        if self.face_cascade is None:
            return [], 0.0, "OpenCV unavailable"
        
        # Validate input frame
        if frame is None or frame.size == 0:
            return [], 0.0, "Invalid frame"
        
        try:
            # Ensure frame is in correct format
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif len(frame.shape) == 2:
                gray = frame.copy()
            else:
                return [], 0.0, "Invalid frame format"
            
            # Apply histogram equalization and denoising with error handling
            try:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray)
                enhanced_gray = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
            except cv2.error:
                enhanced_gray = gray  # Use original if enhancement fails
            
            # Detect frontal faces with safe parameters
            try:
                frontal_faces = self.face_cascade.detectMultiScale(
                    enhanced_gray,
                    scaleFactor=1.05,
                    minNeighbors=6,
                    minSize=(60, 60),
                    maxSize=(400, 400),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            except cv2.error as e:
                print(f"Face detection error: {e}")
                frontal_faces = np.array([])
            
            # Detect profile faces if no frontal faces found
            profile_faces = np.array([])
            if len(frontal_faces) == 0 and self.profile_cascade is not None:
                try:
                    profile_faces = self.profile_cascade.detectMultiScale(
                        enhanced_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(60, 60),
                        maxSize=(400, 400)
                    )
                except cv2.error:
                    profile_faces = np.array([])
            
            # Combine results safely
            all_faces = []
            if len(frontal_faces) > 0:
                all_faces.extend(frontal_faces.tolist())
            if len(profile_faces) > 0:
                all_faces.extend(profile_faces.tolist())
            
            confidence = 0.85 if len(all_faces) > 0 else 0.0
            
            return all_faces, confidence, "OpenCV Enhanced"
            
        except Exception as e:
            print(f"OpenCV face detection error: {e}")
            return [], 0.0, f"OpenCV Error: {str(e)}"

    def calculate_ear_opencv_advanced(self, eye_region):
        """Advanced EAR calculation using contour analysis with enhanced error handling"""
        try:
            # Validate input
            if eye_region is None or eye_region.size == 0:
                return 0.3
                
            # Ensure minimum size
            if eye_region.shape[0] < 5 or eye_region.shape[1] < 5:
                return 0.3
                
            # Convert to grayscale safely
            if len(eye_region.shape) == 3:
                gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            elif len(eye_region.shape) == 2:
                gray_eye = eye_region.copy()
            else:
                return 0.3
            
            # Apply morphological operations to enhance eye features
            try:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                enhanced = cv2.morphologyEx(gray_eye, cv2.MORPH_OPEN, kernel)
                enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            except cv2.error:
                enhanced = gray_eye
            
            # Adaptive thresholding for better edge detection
            try:
                thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 11, 2)
            except cv2.error:
                # Fallback to simple thresholding
                _, thresh = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours with error handling
            try:
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except cv2.error:
                return 0.3
            
            if len(contours) == 0:
                return 0.3
            
            # Find the largest contour (likely the eye opening)
            try:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Validate contour
                if len(largest_contour) < 3:
                    return 0.3
                
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Validate rectangle
                if w <= 0 or h <= 0:
                    return 0.3
                
                # Enhanced EAR calculation
                aspect_ratio = h / w
                
                # Additional metrics with error handling
                contour_area = cv2.contourArea(largest_contour)
                total_area = eye_region.shape[0] * eye_region.shape[1]
                
                if total_area > 0:
                    area_ratio = contour_area / total_area
                else:
                    area_ratio = 0
                
                # Combine metrics for more accurate EAR
                ear = (aspect_ratio * 0.7) + (area_ratio * 0.3)
                
                # Ensure valid EAR value
                if np.isnan(ear) or np.isinf(ear):
                    return 0.3
                
                return max(0.05, min(0.6, ear))
                
            except (ValueError, cv2.error) as e:
                print(f"Contour processing error: {e}")
                return 0.3
            
        except Exception as e:
            print(f"Advanced EAR calculation error: {e}")
            return 0.3

    def detect_eyes_enhanced(self, face_region, frame, face_x, face_y):
        """Enhanced eye detection with multiple methods and array error handling"""
        if self.eye_cascade is None:
            return []
        
        # Validate inputs
        if face_region is None or face_region.size == 0:
            return []
        
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Convert to grayscale safely
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            elif len(face_region.shape) == 2:
                gray_face = face_region.copy()
            else:
                return []
            
            # Apply enhancement with error handling
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                enhanced_face = clahe.apply(gray_face)
            except cv2.error:
                enhanced_face = gray_face
            
            # Detect eyes with multiple parameter sets for robustness
            all_eyes = []
            
            # Try different detection parameters with error handling
            detection_params = [
                {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": (15, 15), "maxSize": (60, 60)},
                {"scaleFactor": 1.05, "minNeighbors": 4, "minSize": (12, 12), "maxSize": (50, 50)},
                {"scaleFactor": 1.2, "minNeighbors": 2, "minSize": (20, 20), "maxSize": (70, 70)}
            ]
            
            for params in detection_params:
                try:
                    eyes = self.eye_cascade.detectMultiScale(enhanced_face, **params)
                    if len(eyes) > 0:
                        all_eyes.extend(eyes.tolist())
                except cv2.error:
                    continue
            
            # Remove duplicates and overlapping detections
            filtered_eyes = self.filter_overlapping_detections(all_eyes)
            
            ear_values = []
            
            for eye_data in filtered_eyes:
                # Validate eye data format
                if len(eye_data) != 4:
                    continue
                
                ex, ey, ew, eh = eye_data
                
                # Validate coordinates
                if ex < 0 or ey < 0 or ew <= 0 or eh <= 0:
                    continue
                
                # Extract eye region with padding and bounds checking
                padding = 5
                eye_y1 = max(0, ey - padding)
                eye_y2 = min(face_region.shape[0], ey + eh + padding)
                eye_x1 = max(0, ex - padding)
                eye_x2 = min(face_region.shape[1], ex + ew + padding)
                
                # Validate region bounds
                if eye_y2 <= eye_y1 or eye_x2 <= eye_x1:
                    continue
                
                eye_region = face_region[eye_y1:eye_y2, eye_x1:eye_x2]
                
                if eye_region.size > 0:
                    ear = self.calculate_ear_opencv_advanced(eye_region)
                    ear_values.append(ear)
                    
                    # Draw eye rectangle with color coding and bounds checking
                    color = (0, 255, 0) if ear > self.EYE_AR_THRESH else (0, 0, 255)
                    thickness = 3 if ear <= self.EYE_AR_THRESH else 2
                    
                    # Calculate absolute coordinates with bounds checking
                    abs_x1 = max(0, min(frame.shape[1] - 1, face_x + ex))
                    abs_y1 = max(0, min(frame.shape[0] - 1, face_y + ey))
                    abs_x2 = max(0, min(frame.shape[1] - 1, face_x + ex + ew))
                    abs_y2 = max(0, min(frame.shape[0] - 1, face_y + ey + eh))
                    
                    # Only draw if coordinates are valid
                    if abs_x2 > abs_x1 and abs_y2 > abs_y1:
                        try:
                            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), color, thickness)
                            
                            # Add EAR text with bounds checking
                            text_x = max(5, abs_x1)
                            text_y = max(15, abs_y1 - 5)
                            cv2.putText(frame, f"{ear:.2f}", (text_x, text_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        except cv2.error:
                            pass  # Skip drawing if coordinates are still invalid
            
            return ear_values
            
        except Exception as e:
            print(f"Enhanced eye detection error: {e}")
            return []

    def filter_overlapping_detections(self, detections, overlap_thresh=0.3):
        """Remove overlapping detections with array error handling"""
        if not detections or len(detections) <= 1:
            return detections
        
        try:
            # Convert to list and validate format
            boxes = []
            for detection in detections:
                if isinstance(detection, (list, tuple, np.ndarray)) and len(detection) >= 4:
                    x, y, w, h = detection[:4]
                    # Validate coordinates
                    if w > 0 and h > 0:
                        boxes.append((int(x), int(y), int(w), int(h)))
            
            if len(boxes) <= 1:
                return boxes
            
            # Calculate areas safely
            areas = []
            for (x, y, w, h) in boxes:
                try:
                    area = w * h
                    if area > 0:
                        areas.append(area)
                    else:
                        areas.append(1)  # Minimum area
                except (ValueError, OverflowError):
                    areas.append(1)
            
            # Sort by area (keep larger detections)
            try:
                indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
            except (IndexError, TypeError):
                return boxes[:1]  # Return first box if sorting fails
            
            keep = []
            while indices:
                current = indices.pop(0)
                keep.append(current)
                
                # Remove overlapping boxes
                remaining_indices = []
                for i in indices:
                    try:
                        overlap = self.calculate_overlap(boxes[current], boxes[i])
                        if overlap < overlap_thresh:
                            remaining_indices.append(i)
                    except (IndexError, ValueError, ZeroDivisionError):
                        continue
                
                indices = remaining_indices
            
            return [boxes[i] for i in keep if i < len(boxes)]
            
        except Exception as e:
            print(f"Filter overlapping detections error: {e}")
            return detections[:1] if detections else []

    def calculate_overlap(self, box1, box2):
        """Calculate overlap ratio between two bounding boxes with error handling"""
        try:
            # Validate input boxes
            if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
                return 0
            
            x1, y1, w1, h1 = box1[:4]
            x2, y2, w2, h2 = box2[:4]
            
            # Validate dimensions
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                return 0
            
            # Calculate intersection
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            
            if xi2 <= xi1 or yi2 <= yi1:
                return 0
            
            intersection = (xi2 - xi1) * (yi2 - yi1)
            union = w1 * h1 + w2 * h2 - intersection
            
            # Prevent division by zero
            if union <= 0:
                return 0
            
            overlap = intersection / union
            
            # Validate result
            if np.isnan(overlap) or np.isinf(overlap):
                return 0
            
            return max(0, min(1, overlap))
            
        except Exception as e:
            print(f"Overlap calculation error: {e}")
            return 0

    def recv(self, frame):
        """Enhanced main processing function with comprehensive error handling"""
        try:
            # Validate input frame
            if frame is None:
                return frame
            
            # Convert frame to array with error handling
            try:
                img = frame.to_ndarray(format="bgr24")
            except Exception as e:
                print(f"Frame conversion error: {e}")
                return frame
            
            # Validate image array
            if img is None or img.size == 0:
                return frame
            
            if len(img.shape) != 3 or img.shape[2] != 3:
                print(f"Invalid image shape: {img.shape}")
                return frame
            
            self.frame_count += 1
            
            # Primary detection using best available method
            faces = []
            confidence = 0.0
            method = "None"
            
            try:
                if MEDIAPIPE_AVAILABLE and self.face_detection is not None:
                    faces, confidence, method = self.detect_face_mediapipe(img)
                    
                    # Use OpenCV as fallback if MediaPipe fails
                    if len(faces) == 0:
                        fallback_faces, fallback_conf, fallback_method = self.detect_face_opencv(img)
                        if len(fallback_faces) > 0:
                            faces, confidence, method = fallback_faces, 0.7, "OpenCV Fallback"
                else:
                    faces, confidence, method = self.detect_face_opencv(img)
            except Exception as e:
                print(f"Face detection error: {e}")
                faces, confidence, method = [], 0.0, "Error"
            
            self.faces_detected = len(faces)
            self.face_confidence = confidence
            self.detection_method = method
            
            # Process hands for phone detection with error handling
            phone_detected = False
            phone_confidence = 0.0
            phone_info = "No analysis"
            self.hands_detected = 0
            
            if MEDIAPIPE_AVAILABLE and self.hands is not None:
                try:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    hands_results = self.hands.process(rgb_img)
                    
                    if hands_results and hasattr(hands_results, 'multi_hand_landmarks') and hands_results.multi_hand_landmarks:
                        self.hands_detected = len(hands_results.multi_hand_landmarks)
                        phone_detected, phone_confidence, phone_info = self.detect_phone_usage(img, hands_results)
                    
                except Exception as e:
                    print(f"Hand detection error: {e}")
                    self.hands_detected = 0
            
            self.phone_detected = phone_detected
            
            # Eye analysis with comprehensive error handling
            ear_avg = 0.3
            
            if len(faces) > 0:
                # Process first (largest) face with validation
                try:
                    face_data = faces[0]
                    if isinstance(face_data, (tuple, list, np.ndarray)) and len(face_data) >= 4:
                        x, y, w, h = face_data[:4]
                        
                        # Convert to integers and validate
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        
                        # Ensure coordinates are valid and within image bounds
                        x = max(0, min(img.shape[1] - 1, x))
                        y = max(0, min(img.shape[0] - 1, y))
                        w = max(1, min(img.shape[1] - x, w))
                        h = max(1, min(img.shape[0] - y, h))
                        
                        x2 = min(img.shape[1], x + w)
                        y2 = min(img.shape[0], y + h)
                        
                        # Validate final coordinates
                        if x2 > x and y2 > y and x >= 0 and y >= 0:
                            # Draw face rectangle with confidence color
                            if confidence > 0.8:
                                face_color = (0, 255, 0)  # Green - high confidence
                            elif confidence > 0.6:
                                face_color = (0, 255, 255)  # Yellow - medium confidence
                            else:
                                face_color = (0, 165, 255)  # Orange - low confidence
                            
                            try:
                                cv2.rectangle(img, (x, y), (x2, y2), face_color, 3)
                                
                                # Add text with bounds checking
                                text_x = max(5, x)
                                text_y = max(25, y - 10)
                                cv2.putText(img, f"Face ({confidence:.2f})", (text_x, text_y), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
                            except cv2.error:
                                pass  # Skip drawing if coordinates are invalid
                            
                            # Extract face region safely
                            if y2 > y and x2 > x:
                                try:
                                    face_region = img[y:y2, x:x2]
                                    
                                    # Validate face region
                                    if face_region.size > 0 and len(face_region.shape) == 3:
                                        # Enhanced eye detection
                                        ear_values = self.detect_eyes_enhanced(face_region, img, x, y)
                                        self.eyes_detected = len(ear_values)
                                        
                                        # Calculate average EAR with smoothing
                                        if len(ear_values) > 0:
                                            ear_avg = sum(ear_values) / len(ear_values)
                                            self.ear_history.append(ear_avg)
                                            
                                            # Apply temporal smoothing
                                            if len(self.ear_history) > 3:
                                                recent_ears = list(self.ear_history)[-5:]
                                                ear_avg = sum(recent_ears) / len(recent_ears)
                                        
                                        # Analyze head pose if MediaPipe is available
                                        if MEDIAPIPE_AVAILABLE and self.face_mesh is not None:
                                            try:
                                                rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                                                mesh_results = self.face_mesh.process(rgb_face)
                                                
                                                if (mesh_results and 
                                                    hasattr(mesh_results, 'multi_face_landmarks') and 
                                                    mesh_results.multi_face_landmarks):
                                                    for landmarks in mesh_results.multi_face_landmarks:
                                                        self.analyze_head_pose(landmarks, face_region.shape[1], face_region.shape[0])
                                            except Exception as e:
                                                print(f"Head pose analysis error: {e}")
                                
                                except Exception as e:
                                    print(f"Face region extraction error: {e}")
                    else:
                        print(f"Invalid face data format: {face_data}")
                        
                except Exception as e:
                    print(f"Face processing error: {e}")
                
                self.current_ear = ear_avg
                
                # Enhanced drowsiness detection
                try:
                    if ear_avg < self.EYE_AR_THRESH:
                        self.EYE_COUNTER += 1
                        
                        if self.EYE_COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                            self.DROWSY_ALARM = True
                            
                            # Enhanced alert visualization with error handling
                            try:
                                cv2.putText(img, "DROWSINESS ALERT!", (10, 60),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                cv2.putText(img, "WAKE UP!", (10, 110),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                
                                # Animated flashing border
                                if self.frame_count % 6 < 3:
                                    border_coords = (5, 5, img.shape[1]-5, img.shape[0]-5)
                                    if all(c >= 0 for c in border_coords):
                                        cv2.rectangle(img, (border_coords[0], border_coords[1]), 
                                                    (border_coords[2], border_coords[3]), (0, 0, 255), 8)
                                
                                # Add severity indicator
                                severity = min(self.EYE_COUNTER - self.EYE_AR_CONSEC_FRAMES, 20)
                                cv2.putText(img, f"Severity: {severity}", (10, 150),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            except cv2.error:
                                pass  # Skip drawing if there's an error
                    else:
                        self.EYE_COUNTER = max(0, self.EYE_COUNTER - 2)  # Faster recovery
                        if self.EYE_COUNTER == 0:
                            self.DROWSY_ALARM = False
                except Exception as e:
                    print(f"Drowsiness detection error: {e}")
            
            # Phone detection alerts with error handling
            try:
                self.phone_history.append(phone_detected)
                if len(self.phone_history) > 0:
                    consistent_phone = sum(self.phone_history) >= len(self.phone_history) * 0.6
                else:
                    consistent_phone = False
                
                if consistent_phone:
                    self.PHONE_COUNTER += 1
                    if self.PHONE_COUNTER >= self.PHONE_CONSEC_FRAMES:
                        self.PHONE_ALARM = True
                else:
                    self.PHONE_COUNTER = max(0, self.PHONE_COUNTER - 1)
                    if self.PHONE_COUNTER == 0:
                        self.PHONE_ALARM = False
            except Exception as e:
                print(f"Phone detection logic error: {e}")
            
            # Phone usage alert with error handling
            if self.PHONE_ALARM:
                try:
                    cv2.putText(img, "PHONE DETECTED!", (10, 200),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                    cv2.putText(img, "PUT PHONE DOWN!", (10, 240),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    
                    # Blue border for phone alert
                    if self.frame_count % 8 < 4:
                        border_w, border_h = img.shape[1], img.shape[0]
                        if border_w > 0 and border_h > 0:
                            cv2.rectangle(img, (0, 0), (border_w, border_h), (255, 0, 0), 6)
                except cv2.error:
                    pass
            
            # Head pose warnings with error handling
            try:
                if hasattr(self, 'head_pose') and self.head_pose.get('looking_down', False):
                    cv2.putText(img, "LOOKING DOWN", (10, 280),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except cv2.error:
                pass
            
            # Enhanced metrics display with error handling
            try:
                metrics_y = max(120, img.shape[0] - 120)
                
                ear_status = 'CLOSED' if ear_avg < self.EYE_AR_THRESH else 'OPEN'
                cv2.putText(img, f"EAR: {ear_avg:.3f} ({ear_status})", 
                           (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(img, f"Eyes: {self.eyes_detected} | Faces: {self.faces_detected}", 
                           (10, metrics_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(img, f"Hands: {self.hands_detected} | Phone: {phone_confidence:.2f}", 
                           (10, metrics_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(img, f"Method: {method}", 
                           (10, metrics_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Frame rate indicator
                frame_text_x = max(120, img.shape[1] - 120)
                cv2.putText(img, f"Frame: {self.frame_count}", 
                           (frame_text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Status indicators in top right
                status_x = max(200, img.shape[1] - 200)
                if self.DROWSY_ALARM:
                    cv2.putText(img, "DROWSY", (status_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif self.PHONE_ALARM:
                    cv2.putText(img, "PHONE", (status_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(img, "ALERT", (status_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
            except cv2.error as e:
                print(f"Text drawing error: {e}")
            
            # Convert back to video frame
            try:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception as e:
                print(f"Frame conversion back error: {e}")
                return frame
            
        except Exception as e:
            print(f"Main processing error: {e}")
            # If processing fails, return original frame with error message
            try:
                img = frame.to_ndarray(format="bgr24")
                error_msg = f"Processing Error: {str(e)[:50]}"
                cv2.putText(img, error_msg, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except:
                return frame


def process_uploaded_image_enhanced(uploaded_file):
    """Enhanced image processing with phone and eye detection and error handling"""
    try:
        # Load image with validation
        if uploaded_file is None:
            return None, None
        
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Validate image array
        if img_array.size == 0:
            st.error("Empty image uploaded")
            return None, None
        
        # Convert RGB to BGR for OpenCV with proper validation
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 2:
            # Convert grayscale to BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            st.error("Unsupported image format")
            return None, None
        
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
            try:
                # Enhanced face detection
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                
                # Apply enhancement with error handling
                try:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced_gray = clahe.apply(gray)
                except cv2.error:
                    enhanced_gray = gray
                
                faces = face_cascade.detectMultiScale(
                    enhanced_gray, 
                    scaleFactor=1.05, 
                    minNeighbors=6, 
                    minSize=(50, 50),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                results["faces"] = len(faces)
                
                # Process each face
                ear_values = []
                for face_data in faces:
                    if len(face_data) < 4:
                        continue
                    
                    x, y, w, h = face_data[:4]
                    
                    # Validate and constrain coordinates
                    x = max(0, min(img_array.shape[1] - 1, int(x)))
                    y = max(0, min(img_array.shape[0] - 1, int(y)))
                    w = max(1, min(img_array.shape[1] - x, int(w)))
                    h = max(1, min(img_array.shape[0] - y, int(h)))
                    
                    x2 = min(img_array.shape[1], x + w)
                    y2 = min(img_array.shape[0], y + h)
                    
                    # Validate final bounds
                    if x2 <= x or y2 <= y:
                        continue
                    
                    # Draw face
                    try:
                        cv2.rectangle(img_array, (x, y), (x2, y2), (0, 255, 0), 3)
                        
                        text_x = max(5, x)
                        text_y = max(25, y - 10)
                        cv2.putText(img_array, "Face Detected", (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except cv2.error:
                        pass
                    
                    # Detect eyes in face region
                    try:
                        face_region = img_array[y:y2, x:x2]
                        
                        if face_region.size > 0 and len(face_region.shape) == 3:
                            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                            
                            if eye_cascade is not None:
                                eyes = eye_cascade.detectMultiScale(
                                    gray_face, 
                                    scaleFactor=1.1, 
                                    minNeighbors=4, 
                                    minSize=(15, 15),
                                    maxSize=(60, 60)
                                )
                                
                                results["eyes"] += len(eyes)
                                
                                for eye_data in eyes:
                                    if len(eye_data) < 4:
                                        continue
                                    
                                    ex, ey, ew, eh = eye_data[:4]
                                    
                                    # Validate eye coordinates
                                    ex = max(0, min(face_region.shape[1] - 1, int(ex)))
                                    ey = max(0, min(face_region.shape[0] - 1, int(ey)))
                                    ew = max(1, min(face_region.shape[1] - ex, int(ew)))
                                    eh = max(1, min(face_region.shape[0] - ey, int(eh)))
                                    
                                    if ex + ew > face_region.shape[1] or ey + eh > face_region.shape[0]:
                                        continue
                                    
                                    # Calculate EAR for this eye
                                    try:
                                        eye_region = face_region[ey:ey + eh, ex:ex + ew]
                                        
                                        if eye_region.size > 0:
                                            # Simple EAR calculation with error handling
                                            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                                            _, thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY)
                                            
                                            white_pixels = cv2.countNonZero(thresh)
                                            total_pixels = thresh.shape[0] * thresh.shape[1]
                                            
                                            if total_pixels > 0:
                                                ear = (white_pixels / total_pixels) * 0.5
                                                ear_values.append(max(0.05, min(0.6, ear)))
                                            
                                            # Draw eye with color based on estimated state
                                            eye_color = (0, 255, 0) if len(ear_values) > 0 and ear_values[-1] > 0.25 else (0, 0, 255)
                                            
                                            # Calculate absolute coordinates for drawing
                                            abs_ex = x + ex
                                            abs_ey = y + ey
                                            abs_ex2 = x + ex + ew
                                            abs_ey2 = y + ey + eh
                                            
                                            # Bounds check for drawing
                                            if (0 <= abs_ex < img_array.shape[1] and 
                                                0 <= abs_ey < img_array.shape[0] and
                                                abs_ex2 <= img_array.shape[1] and 
                                                abs_ey2 <= img_array.shape[0]):
                                                
                                                cv2.rectangle(img_array, (abs_ex, abs_ey), (abs_ex2, abs_ey2), eye_color, 2)
                                                
                                                if len(ear_values) > 0:
                                                    text_x = max(5, abs_ex)
                                                    text_y = max(15, abs_ey - 5)
                                                    cv2.putText(img_array, f"EAR: {ear_values[-1]:.2f}", 
                                                              (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, eye_color, 1)
                                    except Exception as e:
                                        print(f"Eye region processing error: {e}")
                                        continue
                    except Exception as e:
                        print(f"Face region processing error: {e}")
                        continue
                        
            except Exception as e:
                results["analysis"].append(f"Face detection error: {str(e)}")
        
        # Calculate average EAR
        if ear_values:
            results["avg_ear"] = sum(ear_values) / len(ear_values)
            results["analysis"].append(f"Average EAR: {results['avg_ear']:.3f}")
            results["analysis"].append("Eyes appear CLOSED" if results["avg_ear"] < 0.25 else "Eyes appear OPEN")
        
        # Phone detection using MediaPipe hands with error handling
        if MEDIAPIPE_AVAILABLE and detection_result[2] is not None:
            try:
                hands_detector = detection_result[2]
                rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                hands_results = hands_detector.process(rgb_img)
                
                if (hands_results and 
                    hasattr(hands_results, 'multi_hand_landmarks') and 
                    hands_results.multi_hand_landmarks):
                    
                    results["hands"] = len(hands_results.multi_hand_landmarks)
                    
                    # Analyze hand positions for phone usage
                    phone_indicators = 0
                    
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        try:
                            # Draw hand landmarks with bounds checking
                            for landmark in hand_landmarks.landmark:
                                x = int(np.clip(landmark.x * img_array.shape[1], 0, img_array.shape[1] - 1))
                                y = int(np.clip(landmark.y * img_array.shape[0], 0, img_array.shape[0] - 1))
                                cv2.circle(img_array, (x, y), 3, (255, 0, 255), -1)
                            
                            # Check phone holding indicators with bounds checking
                            if len(hand_landmarks.landmark) >= 21:
                                wrist = hand_landmarks.landmark[0]
                                thumb_tip = hand_landmarks.landmark[4]
                                index_tip = hand_landmarks.landmark[8]
                                
                                # Hand near face level
                                if 0.2 < wrist.y < 0.7:
                                    phone_indicators += 1
                                
                                # Thumb and index positioning
                                if thumb_tip.y < wrist.y and index_tip.y < wrist.y:
                                    phone_indicators += 1
                                
                                # Calculate grip distance with error handling
                                try:
                                    grip_dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                                    if 0.05 < grip_dist < 0.3:  # Appropriate phone grip
                                        phone_indicators += 1
                                except (ValueError, OverflowError):
                                    pass
                        except Exception as e:
                            print(f"Hand landmark processing error: {e}")
                    
                    results["phone_confidence"] = min(phone_indicators / 3.0, 1.0)
                    results["analysis"].append(f"Phone usage confidence: {results['phone_confidence']:.2f}")
                    if results["phone_confidence"] > 0.5:
                        results["analysis"].append("⚠️ PHONE USAGE DETECTED")
                        
            except Exception as e:
                results["analysis"].append(f"Hand detection error: {str(e)}")
        
        # Convert back to RGB for display
        try:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        except cv2.error:
            pass
        
        return img_array, results
        
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None, None


# Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})


def main():
    """Enhanced main application with comprehensive driver monitoring"""
    st.set_page_config(
        page_title="🚗 Enhanced Driver Monitoring System",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
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
    .alert-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Enhanced Driver Monitoring System</h1>
        <p>Advanced AI-powered driver safety monitoring with drowsiness and phone usage detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Detection method selection
    detection_method = st.sidebar.selectbox(
        "🔍 Detection Method",
        ["Auto (Best Available)", "MediaPipe Only", "OpenCV Only"],
        help="Choose the detection algorithm to use"
    )
    
    # Sensitivity settings
    st.sidebar.subheader("🎛️ Sensitivity Settings")
    
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
    st.sidebar.subheader("📊 System Status")
    
    # Check system capabilities
    opencv_status = "✅ Available" if cv2 is not None else "❌ Not Available"
    mediapipe_status = "✅ Available" if MEDIAPIPE_AVAILABLE else "❌ Not Available"
    
    st.sidebar.markdown(f"""
    **OpenCV:** {opencv_status}  
    **MediaPipe:** {mediapipe_status}  
    **WebRTC:** ✅ Available
    """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📹 Live Monitoring", "📸 Image Analysis", "ℹ️ Information"])
    
    with tab1:
        st.header("📹 Real-Time Driver Monitoring")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create detection system with user settings
            processor = EnhancedDetectionSystem()
            if hasattr(processor, 'EYE_AR_THRESH'):
                processor.EYE_AR_THRESH = ear_threshold
                processor.EYE_AR_CONSEC_FRAMES = drowsy_frames
            
            # WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="driver-monitoring",
                video_processor_factory=lambda: processor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 640, "ideal": 1280, "max": 1920},
                        "height": {"min": 480, "ideal": 720, "max": 1080},
                        "frameRate": {"min": 15, "ideal": 30, "max": 60}
                    },
                    "audio": False
                },
                async_processing=True,
                desired_playing_state=False
            )
        
        with col2:
            st.subheader("📊 Live Metrics")
            
            # Real-time metrics display
            if webrtc_ctx.video_processor:
                processor = webrtc_ctx.video_processor
                
                # Create metric containers
                ear_container = st.empty()
                detection_container = st.empty()
                alert_container = st.empty()
                status_container = st.empty()
                
                # Update metrics every second
                while webrtc_ctx.state.playing:
                    try:
                        with ear_container.container():
                            ear_value = getattr(processor, 'current_ear', 0.3)
                            ear_status = "CLOSED" if ear_value < ear_threshold else "OPEN"
                            ear_color = "🔴" if ear_value < ear_threshold else "🟢"
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{ear_color} Eye Status: {ear_status}</h4>
                                <p><strong>EAR Value:</strong> {ear_value:.3f}</p>
                                <p><strong>Threshold:</strong> {ear_threshold}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with detection_container.container():
                            faces = getattr(processor, 'faces_detected', 0)
                            eyes = getattr(processor, 'eyes_detected', 0)
                            hands = getattr(processor, 'hands_detected', 0)
                            method = getattr(processor, 'detection_method', 'None')
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>🔍 Detection Stats</h4>
                                <p><strong>Faces:</strong> {faces} | <strong>Eyes:</strong> {eyes}</p>
                                <p><strong>Hands:</strong> {hands}</p>
                                <p><strong>Method:</strong> {method}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with alert_container.container():
                            drowsy_alarm = getattr(processor, 'DROWSY_ALARM', False)
                            phone_alarm = getattr(processor, 'PHONE_ALARM', False)
                            phone_detected = getattr(processor, 'phone_detected', False)
                            
                            if drowsy_alarm:
                                st.markdown("""
                                <div class="danger-box">
                                    <h3>🚨 DROWSINESS ALERT</h3>
                                    <p><strong>Driver appears to be falling asleep!</strong></p>
                                    <p>Immediate attention required</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif phone_alarm:
                                st.markdown("""
                                <div class="danger-box">
                                    <h3>📱 PHONE USAGE ALERT</h3>
                                    <p><strong>Phone usage detected!</strong></p>
                                    <p>Driver should focus on the road</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="success-box">
                                    <h3>✅ System Active</h3>
                                    <p>Monitoring driver behavior</p>
                                    <p>No alerts detected</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with status_container.container():
                            frame_count = getattr(processor, 'frame_count', 0)
                            confidence = getattr(processor, 'face_confidence', 0.0)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>⚡ System Performance</h4>
                                <p><strong>Frames Processed:</strong> {frame_count}</p>
                                <p><strong>Face Confidence:</strong> {confidence:.2f}</p>
                                <p><strong>Status:</strong> {'🟢 Running' if webrtc_ctx.state.playing else '🔴 Stopped'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        time.sleep(1)  # Update every second
                        
                    except Exception as e:
                        st.error(f"Metrics update error: {e}")
                        break
    
    with tab2:
        st.header("📸 Static Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload an image for analysis",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a driver photo to analyze drowsiness and phone usage"
        )
        
        if uploaded_file is not None:
            # Process image
            with st.spinner("Analyzing image..."):
                processed_img, analysis_results = process_uploaded_image_enhanced(uploaded_file)
            
            if processed_img is not None and analysis_results is not None:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📊 Analysis Results")
                    st.image(processed_img, caption="Processed Image with Detections", use_column_width=True)
                
                with col2:
                    st.subheader("📈 Detection Metrics")
                    
                    # Display results
                    faces = analysis_results.get("faces", 0)
                    eyes = analysis_results.get("eyes", 0)
                    hands = analysis_results.get("hands", 0)
                    phone_conf = analysis_results.get("phone_confidence", 0.0)
                    avg_ear = analysis_results.get("avg_ear", 0.3)
                    
                    st.metric("Faces Detected", faces)
                    st.metric("Eyes Detected", eyes)
                    st.metric("Hands Detected", hands)
                    st.metric("Average EAR", f"{avg_ear:.3f}")
                    st.metric("Phone Confidence", f"{phone_conf:.2f}")
                    
                    # Status assessment
                    if avg_ear < 0.25:
                        st.error("🚨 Eyes appear CLOSED - Potential drowsiness")
                    else:
                        st.success("✅ Eyes appear OPEN - Driver alert")
                    
                    if phone_conf > 0.5:
                        st.error("📱 Phone usage detected")
                    else:
                        st.success("✅ No phone usage detected")
                    
                    # Detailed analysis
                    if analysis_results.get("analysis"):
                        st.subheader("🔍 Detailed Analysis")
                        for item in analysis_results["analysis"]:
                            st.write(f"• {item}")
    
    with tab3:
        st.header("ℹ️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Detection Features")
            st.markdown("""
            **Drowsiness Detection:**
            - Eye Aspect Ratio (EAR) monitoring
            - Consecutive frame analysis
            - Multiple detection algorithms
            - Real-time alerting
            
            **Phone Usage Detection:**
            - Hand pose analysis
            - Grip pattern recognition
            - Head pose evaluation
            - Multi-indicator scoring
            
            **Face Detection:**
            - MediaPipe AI models
            - OpenCV Haar cascades
            - Profile face detection
            - Confidence scoring
            """)
        
        with col2:
            st.subheader("⚡ Technical Specifications")
            st.markdown("""
            **Algorithms Used:**
            - MediaPipe Face Detection
            - MediaPipe Face Mesh
            - MediaPipe Hands
            - OpenCV Haar Cascades
            - Advanced image processing
            
            **Performance:**
            - Real-time processing (15-30 FPS)
            - Multi-threaded architecture
            - Adaptive thresholding
            - Error recovery systems
            
            **Supported Formats:**
            - Live webcam feed
            - Static image analysis
            - JPG, PNG, BMP formats
            - HD video processing
            """)
        
        st.subheader("🔧 Troubleshooting")
        
        with st.expander("Common Issues & Solutions"):
            st.markdown("""
            **Camera not working:**
            - Check browser permissions
            - Ensure camera is not used by other apps
            - Try refreshing the page
            
            **Poor detection accuracy:**
            - Ensure good lighting
            - Position face clearly in frame
            - Adjust sensitivity settings
            - Check for camera focus
            
            **Performance issues:**
            - Lower video resolution
            - Close other browser tabs
            - Check system resources
            - Use Chrome for best performance
            
            **MediaPipe errors:**
            - System will fallback to OpenCV
            - Check console for detailed errors
            - Restart the application if needed
            """)
        
        st.subheader("📋 Usage Guidelines")
        
        with st.expander("How to Use This System"):
            st.markdown("""
            **Live Monitoring:**
            1. Click "Start" to begin video monitoring
            2. Position yourself in front of the camera
            3. Ensure good lighting conditions
            4. Monitor alerts and metrics in real-time
            5. Adjust sensitivity settings as needed
            
            **Image Analysis:**
            1. Upload a clear driver photo
            2. Wait for processing to complete
            3. Review detection results
            4. Check EAR values and phone indicators
            
            **Best Practices:**
            - Use in well-lit environments
            - Keep camera clean and focused
            - Position camera at eye level
            - Minimize background distractions
            - Test system before actual use
            """)
        
        # System information
        st.subheader("🖥️ System Information")
        
        system_info = {
            "OpenCV Version": cv2.__version__ if cv2 is not None else "Not Available",
            "MediaPipe Status": "Available" if MEDIAPIPE_AVAILABLE else "Not Available",
            "Streamlit Version": st.__version__,
            "Detection Methods": "MediaPipe + OpenCV" if MEDIAPIPE_AVAILABLE else "OpenCV Only"
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")


if __name__ == "__main__":
    main()
