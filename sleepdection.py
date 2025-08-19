import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading
import time
import pygame
from scipy.spatial import distance as dist
import math
import tempfile
import os
from PIL import Image
import io
import base64


# Load OpenCV cascades
@st.cache_resource
def load_detectors():
    """Load OpenCV face, eye, and other detectors"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    return face_cascade, eye_cascade, profile_cascade


class DrowsinessDetector:
    def __init__(self):
        self.EYE_AR_THRESH = 0.15
        self.EYE_AR_CONSEC_FRAMES = 15
        self.COUNTER = 0
        self.ALARM_ON = False
        self.MOBILE_COUNTER = 0
        self.MOBILE_THRESH = 10
        self.MOBILE_ALERT = False
        
        self.face_cascade, self.eye_cascade, self.profile_cascade = load_detectors()
        
        # Initialize pygame for sound
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.create_alarm_sounds()
        except Exception as e:
            st.warning(f"Audio system not available: {e}")

    def create_alarm_sounds(self):
        """Create different alarm sounds"""
        try:
            sample_rate = 22050
            
            # Drowsiness alarm - lower frequency beep
            self.create_sound_file('drowsy_alarm.wav', 400, 0.8, sample_rate)
            
            # Mobile alert - higher frequency beep
            self.create_sound_file('mobile_alarm.wav', 800, 0.5, sample_rate)
            
        except Exception as e:
            print(f"Error creating alarm sounds: {e}")

    def create_sound_file(self, filename, frequency, duration, sample_rate):
        """Create a specific sound file"""
        frames = int(duration * sample_rate)
        arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
        arr = (arr * 32767 * 0.3).astype(np.int16)
        arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        import wave
        with wave.open(temp_file.name, 'w') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(arr.tobytes())
        
        if filename == 'drowsy_alarm.wav':
            self.drowsy_alarm_file = temp_file.name
        else:
            self.mobile_alarm_file = temp_file.name

    def eye_aspect_ratio_improved(self, eye_region):
        """Improved Eye Aspect Ratio calculation"""
        if eye_region.size == 0:
            return 0.3
        
        # Convert to grayscale if needed
        if len(eye_region.shape) == 3:
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_eye = eye_region
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_eye, (3, 3), 0)
        
        # Use adaptive thresholding for better results
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate the ratio of white pixels (open area) to total pixels
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        
        if total_pixels > 0:
            open_ratio = white_pixels / total_pixels
            # Convert to EAR-like scale (lower values = more closed)
            ear = open_ratio * 0.4  # Scale factor
            return max(0.05, min(0.4, ear))  # Clamp between reasonable bounds
        
        return 0.2

    def detect_eyes_improved(self, face_region, frame, face_x, face_y):
        """Improved eye detection with better EAR calculation"""
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        # Detect eyes in the face region
        eyes = self.eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(15, 15),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        ear_values = []
        eye_centers = []
        
        # Sort eyes by x-coordinate to get left and right eye
        eyes_sorted = sorted(eyes, key=lambda x: x[0])
        
        for i, (ex, ey, ew, eh) in enumerate(eyes_sorted):
            # Add padding around eye region
            padding = 5
            ey_start = max(0, ey - padding)
            ey_end = min(face_region.shape[0], ey + eh + padding)
            ex_start = max(0, ex - padding)
            ex_end = min(face_region.shape[1], ex + ew + padding)
            
            eye_region = face_region[ey_start:ey_end, ex_start:ex_end]
            
            if eye_region.size > 0:
                # Calculate EAR for this eye
                ear = self.eye_aspect_ratio_improved(eye_region)
                ear_values.append(ear)
                
                # Calculate eye center for drawing
                eye_center_x = face_x + ex + ew // 2
                eye_center_y = face_y + ey + eh // 2
                eye_centers.append((eye_center_x, eye_center_y))
                
                # Draw eye rectangle with different colors for left/right
                color = (0, 255, 0) if i == 0 else (0, 255, 255)  # Green for left, Cyan for right
                cv2.rectangle(frame, (face_x + ex, face_y + ey),
                              (face_x + ex + ew, face_y + ey + eh), color, 2)
                
                # Draw eye center
                cv2.circle(frame, (eye_center_x, eye_center_y), 2, (255, 255, 255), -1)
                
                # Add EAR text near eye
                cv2.putText(frame, f"{ear:.2f}", (face_x + ex, face_y + ey - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return ear_values, eye_centers

    def detect_mobile_usage(self, frame, face_region, face_x, face_y, face_w, face_h):
        """Detect potential mobile phone usage"""
        mobile_detected = False
        
        # Define region of interest around the ear area (right side of face)
        ear_region_right = frame[face_y:face_y + face_h, face_x + int(face_w * 0.7):face_x + face_w + 50]
        ear_region_left = frame[face_y:face_y + face_h, max(0, face_x - 50):face_x + int(face_w * 0.3)]
        
        # Convert to HSV for better color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for dark objects (typical phone colors)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        
        # Create mask for dark objects
        mask = cv2.inRange(hsv_frame, lower_dark, upper_dark)
        
        # Check for rectangular dark objects near ear regions
        ear_right_roi = mask[
            face_y:face_y + face_h, face_x + int(face_w * 0.8):min(frame.shape[1], face_x + face_w + 60)]
        ear_left_roi = mask[face_y:face_y + face_h, max(0, face_x - 60):face_x + int(face_w * 0.2)]
        
        # Count dark pixels in ear regions
        right_dark_pixels = cv2.countNonZero(ear_right_roi) if ear_right_roi.size > 0 else 0
        left_dark_pixels = cv2.countNonZero(ear_left_roi) if ear_left_roi.size > 0 else 0
        
        # Threshold for mobile detection (adjust as needed)
        mobile_threshold = 100
        
        if right_dark_pixels > mobile_threshold or left_dark_pixels > mobile_threshold:
            mobile_detected = True
            
            # Draw warning rectangle
            if right_dark_pixels > mobile_threshold:
                cv2.rectangle(frame, (face_x + int(face_w * 0.8), face_y),
                              (min(frame.shape[1], face_x + face_w + 60), face_y + face_h), (0, 0, 255), 2)
                cv2.putText(frame, "PHONE?", (face_x + face_w - 20, face_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if left_dark_pixels > mobile_threshold:
                cv2.rectangle(frame, (max(0, face_x - 60), face_y),
                              (face_x + int(face_w * 0.2), face_y + face_h), (0, 0, 255), 2)
                cv2.putText(frame, "PHONE?", (face_x - 60, face_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return mobile_detected

    def process_frame(self, frame):
        """Process a single frame for drowsiness and mobile detection"""
        frame_height, frame_width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        ear_avg = 0.3  # Default EAR value
        drowsy_status = "😊 Driver Alert & Focused"
        mobile_status = "📱 No Phone Detected"
        mobile_detected = False
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Driver", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Extract face region
            face_region = frame[y:y + h, x:x + w]
            
            # Detect eyes in face region
            ear_values, eye_centers = self.detect_eyes_improved(face_region, frame, x, y)
            
            # Detect mobile phone usage
            mobile_detected = self.detect_mobile_usage(frame, face_region, x, y, w, h)
            
            if len(ear_values) >= 2:
                ear_avg = sum(ear_values) / len(ear_values)
            elif len(ear_values) == 1:
                ear_avg = ear_values[0]
            
            # Check for drowsiness
            if ear_avg < self.EYE_AR_THRESH:
                self.COUNTER += 1
                
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    if not self.ALARM_ON:
                        self.ALARM_ON = True
                    
                    drowsy_status = "😴 DROWSINESS ALERT - WAKE UP!"
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, "EYES CLOSED - WAKE UP!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Eyes are open - reset counter and stop alarm
                self.COUNTER = 0
                if self.ALARM_ON:
                    self.ALARM_ON = False
            
            # Handle mobile detection
            if mobile_detected:
                self.MOBILE_COUNTER += 1
                if self.MOBILE_COUNTER >= self.MOBILE_THRESH:
                    if not self.MOBILE_ALERT:
                        self.MOBILE_ALERT = True
                    
                    mobile_status = "📱 MOBILE PHONE DETECTED!"
                    cv2.putText(frame, "AVOID USING MOBILE!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, "STOP CAR & USE PHONE", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                self.MOBILE_COUNTER = max(0, self.MOBILE_COUNTER - 1)
                if self.MOBILE_COUNTER == 0:
                    self.MOBILE_ALERT = False
            
            # Display metrics on frame
            cv2.putText(frame, f"EAR: {ear_avg:.3f}", (frame_width - 150, frame_height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Eyes: {len(ear_values)}", (frame_width - 150, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Threshold: {self.EYE_AR_THRESH:.3f}",
                        (frame_width - 150, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            break  # Process only the first face
        
        return frame, ear_avg, len(ear_values), drowsy_status, mobile_status, mobile_detected

    def play_drowsy_alarm(self):
        """Play drowsiness alarm sound"""
        try:
            if hasattr(self, 'drowsy_alarm_file') and os.path.exists(self.drowsy_alarm_file):
                pygame.mixer.music.load(self.drowsy_alarm_file)
                pygame.mixer.music.play(-1)  # Loop indefinitely
        except Exception as e:
            print(f"Error playing drowsy alarm: {e}")

    def play_mobile_alarm(self):
        """Play mobile usage alarm sound"""
        try:
            if hasattr(self, 'mobile_alarm_file') and os.path.exists(self.mobile_alarm_file):
                pygame.mixer.music.load(self.mobile_alarm_file)
                pygame.mixer.music.play(2)  # Play 2 times
        except Exception as e:
            print(f"Error playing mobile alarm: {e}")

    def stop_alarm(self):
        """Stop all alarm sounds"""
        try:
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"Error stopping alarm: {e}")


def main():
    # Custom CSS for modern UI
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .alert-card {
        background: linear-gradient(135deg, #ff4757 0%, #ff3838 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff2f2f;
        color: white;
        margin: 1rem 0;
        animation: pulse 1.5s infinite;
        font-weight: bold;
    }

    .mobile-alert-card {
        background: linear-gradient(135deg, #ff6348 0%, #ff4757 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff3838;
        color: white;
        margin: 1rem 0;
        animation: shake 1s infinite;
        font-weight: bold;
    }

    .status-good {
        background: linear-gradient(135deg, #2ed573 0%, #7bed9f 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00d4aa;
        color: white;
        font-weight: bold;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }

    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚨 AI Driver Safety Monitoring System</h1>
        <p>Advanced Drowsiness Detection + Mobile Phone Usage Alert</p>
        <small>Powered by OpenCV & TensorFlow - Real-time Safety Monitoring</small>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = DrowsinessDetector()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙ Control Panel")

        sensitivity = st.slider("👁 Drowsiness Sensitivity", 0.05, 0.25, 0.15, 0.01)
        st.session_state.detector.EYE_AR_THRESH = sensitivity

        frame_threshold = st.slider("⏱ Drowsiness Alert Frame Threshold", 5, 30, 15, 5)
        st.session_state.detector.EYE_AR_CONSEC_FRAMES = frame_threshold

        mobile_threshold = st.slider("📱 Mobile Detection Threshold", 5, 20, 10, 1)
        st.session_state.detector.MOBILE_THRESH = mobile_threshold

        st.markdown("### 🎵 Audio Settings")
        enable_drowsy_sound = st.checkbox("Enable Drowsiness Alerts", True)
        enable_mobile_sound = st.checkbox("Enable Mobile Phone Alerts", True)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📹 Camera Input")
        
        # Use Streamlit's camera_input for webcam access
        picture = st.camera_input("Take a picture to analyze for drowsiness")
        
        if picture is not None:
            # Convert the uploaded image to OpenCV format
            image = Image.open(picture)
            frame = np.array(image)
            
            # Process the frame
            processed_frame, ear_avg, eyes_detected, drowsy_status, mobile_status, mobile_detected = st.session_state.detector.process_frame(frame)
            
            # Display the processed frame
            st.image(processed_frame, channels="RGB", use_column_width=True)
            
            # Play alarms if needed
            if st.session_state.detector.ALARM_ON and enable_drowsy_sound:
                st.session_state.detector.play_drowsy_alarm()
                st.audio(st.session_state.detector.drowsy_alarm_file if hasattr(st.session_state.detector, 'drowsy_alarm_file') else None)
            
            if st.session_state.detector.MOBILE_ALERT and enable_mobile_sound:
                st.session_state.detector.play_mobile_alarm()
                st.audio(st.session_state.detector.mobile_alarm_file if hasattr(st.session_state.detector, 'mobile_alarm_file') else None)
        
        else:
            st.info("📸 Click 'Take photo' to capture and analyze an image from your camera")

    with col2:
        st.markdown("### 📈 Status Dashboard")
        
        if picture is not None:
            # Status display
            if st.session_state.detector.ALARM_ON:
                st.markdown('''
                <div class="alert-card">
                    <h3>😴 DROWSINESS ALERT!</h3>
                    <p>Driver appears drowsy - please pull over safely</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="status-good">
                    <h4>😊 Driver Alert & Focused</h4>
                </div>
                ''', unsafe_allow_html=True)
            
            # Mobile phone status
            if st.session_state.detector.MOBILE_ALERT:
                st.markdown('''
                <div class="mobile-alert-card">
                    <h3>📱 MOBILE PHONE ALERT!</h3>
                    <p>Please stop your car safely and then use your phone</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="status-good">
                    <h4>📱 Hands Free - Good Driving</h4>
                </div>
                ''', unsafe_allow_html=True)
            
            # Metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f'''
                <div class="metric-card">
                    <h4>Average EAR</h4>
                    <h2>{ear_avg:.3f}</h2>
                    <small>Threshold: {st.session_state.detector.EYE_AR_THRESH:.3f}</small>
                </div>
                ''', unsafe_allow_html=True)

            with col_m2:
                st.markdown(f'''
                <div class="metric-card">
                    <h4>Eyes Detected</h4>
                    <h2>{eyes_detected}</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            # Additional metrics
            col_m3, col_m4 = st.columns(2)
            with col_m3:
                st.markdown(f'''
                <div class="metric-card">
                    <h4>Closed Eye Frames</h4>
                    <h2>{st.session_state.detector.COUNTER}</h2>
                    <small>Max: {st.session_state.detector.EYE_AR_CONSEC_FRAMES}</small>
                </div>
                ''', unsafe_allow_html=True)

            with col_m4:
                st.markdown(f'''
                <div class="metric-card">
                    <h4>Mobile Detection</h4>
                    <h2>{st.session_state.detector.MOBILE_COUNTER}</h2>
                    <small>Threshold: {st.session_state.detector.MOBILE_THRESH}</small>
                </div>
                ''', unsafe_allow_html=True)

    # Alternative: Video upload option
    st.markdown("### 🎥 Or Upload a Video File")
    uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_video is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        # Process video
        cap = cv2.VideoCapture(tfile.name)
        
        if cap.isOpened():
            st.video(uploaded_video)
            
            if st.button("🔍 Analyze Video"):
                progress_bar = st.progress(0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                processed_frames = []
                current_frame = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame, ear_avg, eyes_detected, drowsy_status, mobile_status, mobile_detected = st.session_state.detector.process_frame(frame)
                    processed_frames.append(processed_frame)
                    
                    current_frame += 1
                    progress_bar.progress(current_frame / frame_count)
                    
                    # Process only every 10th frame for performance
                    for _ in range(9):
                        cap.read()
                        current_frame += 1
                        if current_frame >= frame_count:
                            break
                
                st.success(f"✅ Processed {len(processed_frames)} frames")
                
                if processed_frames:
                    st.image(processed_frames[0], caption="First processed frame", use_column_width=True)
        
        cap.release()
        os.unlink(tfile.name)

    # Information section
    with st.expander("ℹ How It Works - Camera Input Version"):
        st.markdown('''
        ### 📸 Camera Input Method
        This version uses Streamlit's built-in camera input component instead of direct webcam access:
        
        1. **Click "Take photo"** to capture an image from your webcam
        2. **Real-time Analysis** happens on each captured frame
        3. **Audio Alerts** play when drowsiness or mobile usage is detected
        4. **Video Upload** option for analyzing pre-recorded videos
        
        ### 🔧 Key Features
        - **Streamlit Camera Input**: Web-safe camera access
        - **Single Frame Analysis**: Process individual photos
        - **Video File Support**: Upload and analyze video files
        - **Audio Alerts**: Sound notifications for safety alerts
        - **Real-time Metrics**: Comprehensive status dashboard
        
        ### 🚗 Usage Instructions
        1. **Allow camera access** when prompted by your browser
        2. **Take photos** regularly while driving (when stopped safely)
        3. **Monitor alerts** on the dashboard
        4. **Upload videos** for batch analysis
        
        ### 📱 Mobile Phone Detection
        - Analyzes areas around ears for dark rectangular objects
        - Triggers alerts when phone-like objects are detected
        - Recommends stopping the car before using phone
        
        ### ⚠ Safety Note
        This photo-based version is more suitable for:
        - Testing the detection algorithms
        - Analyzing recorded driving footage
        - Educational purposes
        
        For continuous real-time monitoring, consider using the system with a dedicated camera setup.
        ''')


if __name__ == "__main__":
    # Page configuration
    st.set_page_config(
        page_title="AI Driver Safety Monitor",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    main()
