import streamlit as st
import cv2
import numpy as np
import threading
import time
import tempfile
import os
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import pygame
from scipy.spatial import distance as dist


# Load OpenCV cascades
@st.cache_resource
def load_detectors():
    """Load OpenCV face, eye, and other detectors"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    return face_cascade, eye_cascade, profile_cascade


class DrowsinessVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.EYE_AR_THRESH = 0.15
        self.EYE_AR_CONSEC_FRAMES = 15
        self.COUNTER = 0
        self.ALARM_ON = False
        self.MOBILE_COUNTER = 0
        self.MOBILE_THRESH = 10
        self.MOBILE_ALERT = False
        
        self.face_cascade, self.eye_cascade, self.profile_cascade = load_detectors()
        
        # Shared state for metrics
        self.current_ear = 0.3
        self.eyes_detected = 0
        self.faces_detected = 0
        
        # Initialize pygame for sound
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.create_alarm_sounds()
        except Exception as e:
            print(f"Audio system not available: {e}")

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
        
        # Threshold for mobile detection
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

    def transform(self, frame):
        """Main video processing function called for each frame"""
        img = frame.to_ndarray(format="bgr24")
        
        frame_height, frame_width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        self.faces_detected = len(faces)
        ear_avg = 0.3  # Default EAR value
        mobile_detected = False
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Driver", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Extract face region
            face_region = img[y:y + h, x:x + w]
            
            # Detect eyes in face region
            ear_values, eye_centers = self.detect_eyes_improved(face_region, img, x, y)
            self.eyes_detected = len(ear_values)
            
            # Detect mobile phone usage
            mobile_detected = self.detect_mobile_usage(img, face_region, x, y, w, h)
            
            if len(ear_values) >= 2:
                ear_avg = sum(ear_values) / len(ear_values)
            elif len(ear_values) == 1:
                ear_avg = ear_values[0]
            
            self.current_ear = ear_avg
            
            # Check for drowsiness
            if ear_avg < self.EYE_AR_THRESH:
                self.COUNTER += 1
                
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    if not self.ALARM_ON:
                        self.ALARM_ON = True
                        try:
                            self.play_drowsy_alarm()
                        except:
                            pass
                    
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(img, "EYES CLOSED - WAKE UP!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Eyes are open - reset counter
                self.COUNTER = 0
                if self.ALARM_ON:
                    self.ALARM_ON = False
            
            # Handle mobile detection
            if mobile_detected:
                self.MOBILE_COUNTER += 1
                if self.MOBILE_COUNTER >= self.MOBILE_THRESH:
                    if not self.MOBILE_ALERT:
                        self.MOBILE_ALERT = True
                        try:
                            self.play_mobile_alarm()
                        except:
                            pass
                    
                    cv2.putText(img, "AVOID USING MOBILE!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(img, "STOP CAR & USE PHONE", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                self.MOBILE_COUNTER = max(0, self.MOBILE_COUNTER - 1)
                if self.MOBILE_COUNTER == 0:
                    self.MOBILE_ALERT = False
            
            # Display metrics on frame
            cv2.putText(img, f"EAR: {ear_avg:.3f}", (frame_width - 150, frame_height - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Eyes: {len(ear_values)}", (frame_width - 150, frame_height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"Threshold: {self.EYE_AR_THRESH:.3f}",
                        (frame_width - 150, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(img, f"Status: {'DROWSY' if self.ALARM_ON else 'ALERT'}",
                        (frame_width - 150, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if self.ALARM_ON else (0, 255, 0), 1)
            
            break  # Process only the first face
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def play_drowsy_alarm(self):
        """Play drowsiness alarm sound"""
        try:
            if hasattr(self, 'drowsy_alarm_file') and os.path.exists(self.drowsy_alarm_file):
                pygame.mixer.music.load(self.drowsy_alarm_file)
                pygame.mixer.music.play(1)  # Play once
        except Exception as e:
            print(f"Error playing drowsy alarm: {e}")

    def play_mobile_alarm(self):
        """Play mobile usage alarm sound"""
        try:
            if hasattr(self, 'mobile_alarm_file') and os.path.exists(self.mobile_alarm_file):
                pygame.mixer.music.load(self.mobile_alarm_file)
                pygame.mixer.music.play(1)  # Play once
        except Exception as e:
            print(f"Error playing mobile alarm: {e}")


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
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚨 Live Driver Safety Monitoring System</h1>
        <p>Real-time Drowsiness Detection + Mobile Phone Usage Alert</p>
        <small>Powered by WebRTC Live Video Streaming</small>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for transformer
    if 'transformer' not in st.session_state:
        st.session_state.transformer = DrowsinessVideoTransformer()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙ Control Panel")

        sensitivity = st.slider("👁 Drowsiness Sensitivity", 0.05, 0.25, 0.15, 0.01)
        st.session_state.transformer.EYE_AR_THRESH = sensitivity

        frame_threshold = st.slider("⏱ Drowsiness Alert Frame Threshold", 5, 30, 15, 5)
        st.session_state.transformer.EYE_AR_CONSEC_FRAMES = frame_threshold

        mobile_threshold = st.slider("📱 Mobile Detection Threshold", 5, 20, 10, 1)
        st.session_state.transformer.MOBILE_THRESH = mobile_threshold

        st.markdown("### 🎵 Audio Settings")
        enable_audio = st.checkbox("Enable Audio Alerts", True)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📹 Live Video Stream")
        
        # WebRTC video streamer
        webrtc_ctx = webrtc_streamer(
            key="driver-safety-monitor",
            video_transformer_factory=lambda: st.session_state.transformer,
            client_settings=ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            ),
            async_processing=True,
        )

    with col2:
        st.markdown("### 📈 Live Status Dashboard")
        
        # Real-time metrics placeholder
        metrics_placeholder = st.empty()
        
        # Update metrics in real-time
        if webrtc_ctx.video_transformer:
            transformer = webrtc_ctx.video_transformer
            
            # Status display
            if transformer.ALARM_ON:
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
            if transformer.MOBILE_ALERT:
                st.markdown('''
                <div class="alert-card">
                    <h3>📱 MOBILE PHONE ALERT!</h3>
                    <p>Please stop your car safely and then use your phone</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Live metrics
            with metrics_placeholder.container():
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>Live EAR</h4>
                        <h2>{transformer.current_ear:.3f}</h2>
                        <small>Threshold: {transformer.EYE_AR_THRESH:.3f}</small>
                    </div>
                    ''', unsafe_allow_html=True)

                with col_m2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>Eyes Detected</h4>
                        <h2>{transformer.eyes_detected}</h2>
                    </div>
                    ''', unsafe_allow_html=True)
                
                col_m3, col_m4 = st.columns(2)
                with col_m3:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>Closed Eye Count</h4>
                        <h2>{transformer.COUNTER}</h2>
                        <small>Max: {transformer.EYE_AR_CONSEC_FRAMES}</small>
                    </div>
                    ''', unsafe_allow_html=True)

                with col_m4:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>Mobile Detection</h4>
                        <h2>{transformer.MOBILE_COUNTER}</h2>
                        <small>Threshold: {transformer.MOBILE_THRESH}</small>
                    </div>
                    ''', unsafe_allow_html=True)

    # Information section
    with st.expander("ℹ How to Use - Live Video Version"):
        st.markdown('''
        ### 🚀 Setup Instructions
        
        1. **Install streamlit-webrtc**:
        ```bash
        pip install streamlit-webrtc
        ```
        
        2. **Allow camera access** when prompted by your browser
        
        3. **Click START** to begin live video processing
        
        4. **Monitor the dashboard** for real-time alerts
        
        ### 🔧 Features
        - **Live Video Stream**: Real-time webcam processing
        - **WebRTC Technology**: Direct browser-to-browser communication
        - **Real-time Alerts**: Instant drowsiness and mobile detection
        - **Audio Notifications**: Sound alerts for safety
        - **Adjustable Parameters**: Fine-tune detection sensitivity
        
        ### 📱 Detection Capabilities
        - **Eye Aspect Ratio (EAR)**: Measures eye openness
        - **Consecutive Frame Counting**: Prevents false positives
        - **Mobile Phone Detection**: Detects objects near ears
        - **Multi-face Support**: Processes multiple faces
        
        ### ⚠ Browser Compatibility
        - **Chrome**: Full support ✅
        - **Firefox**: Full support ✅
        - **Safari**: Limited support ⚠️
        - **Edge**: Full support ✅
        
        ### 🚗 Safety Recommendations
        - Use with proper vehicle mounting
        - Ensure good lighting conditions
        - Regular calibration based on environment
        - Pull over safely when alerts trigger
        ''')


if __name__ == "__main__":
    # Page configuration
    st.set_page_config(
        page_title="Live Driver Safety Monitor",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    main()
