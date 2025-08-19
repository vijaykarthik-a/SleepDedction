import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import threading
import queue
import time


# Load OpenCV cascades
@st.cache_resource
def load_detectors():
    """Load OpenCV face and eye detectors"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        return face_cascade, eye_cascade
    except Exception as e:
        st.error(f"Error loading OpenCV cascades: {e}")
        return None, None


class DrowsinessDetector(VideoTransformerBase):
    def __init__(self):
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 20
        self.COUNTER = 0
        self.ALARM_ON = False
        self.MOBILE_COUNTER = 0
        self.MOBILE_THRESH = 15
        self.MOBILE_ALERT = False
        
        # Load detectors
        self.face_cascade, self.eye_cascade = load_detectors()
        
        # Metrics for dashboard
        self.current_ear = 0.3
        self.eyes_detected = 0
        self.faces_detected = 0
        self.frame_count = 0

    def eye_aspect_ratio(self, eye_region):
        """Calculate Eye Aspect Ratio from eye region"""
        if eye_region.size == 0:
            return 0.3
            
        # Convert to grayscale if needed
        if len(eye_region.shape) == 3:
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_eye = eye_region
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_eye, (3, 3), 0)
        
        # Use Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate ratio of white pixels (open area) to total pixels
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        
        if total_pixels > 0:
            open_ratio = white_pixels / total_pixels
            # Scale to EAR-like values
            ear = open_ratio * 0.5
            return max(0.05, min(0.5, ear))
        
        return 0.3

    def detect_eyes_and_calculate_ear(self, face_region, frame, face_x, face_y):
        """Detect eyes and calculate EAR"""
        if self.eye_cascade is None:
            return [], []
            
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        ear_values = []
        
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            # Extract eye region with padding
            padding = 3
            ey_start = max(0, ey - padding)
            ey_end = min(face_region.shape[0], ey + eh + padding)
            ex_start = max(0, ex - padding)
            ex_end = min(face_region.shape[1], ex + ew + padding)
            
            eye_region = face_region[ey_start:ey_end, ex_start:ex_end]
            
            if eye_region.size > 0:
                # Calculate EAR
                ear = self.eye_aspect_ratio(eye_region)
                ear_values.append(ear)
                
                # Draw eye rectangle
                color = (0, 255, 0) if i == 0 else (255, 255, 0)
                cv2.rectangle(frame, (face_x + ex, face_y + ey),
                            (face_x + ex + ew, face_y + ey + eh), color, 2)
                
                # Display EAR value
                cv2.putText(frame, f"EAR: {ear:.2f}", (face_x + ex, face_y + ey - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return ear_values, eyes

    def detect_mobile_phone(self, frame, face_x, face_y, face_w, face_h):
        """Simple mobile phone detection based on dark objects near ears"""
        mobile_detected = False
        
        # Define ear regions (left and right side of face)
        ear_width = int(face_w * 0.3)
        ear_height = int(face_h * 0.6)
        
        # Right ear region
        right_ear_x = face_x + face_w
        right_ear_y = face_y + int(face_h * 0.2)
        
        # Left ear region  
        left_ear_x = max(0, face_x - ear_width)
        left_ear_y = face_y + int(face_h * 0.2)
        
        # Convert to HSV for better dark object detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for dark objects (phones are usually dark)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 60])
        
        mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Check right ear region
        if right_ear_x + ear_width < frame.shape[1] and right_ear_y + ear_height < frame.shape[0]:
            right_roi = mask[right_ear_y:right_ear_y + ear_height, 
                           right_ear_x:right_ear_x + ear_width]
            right_dark_pixels = cv2.countNonZero(right_roi) if right_roi.size > 0 else 0
            
            if right_dark_pixels > 200:  # Threshold for mobile detection
                mobile_detected = True
                cv2.rectangle(frame, (right_ear_x, right_ear_y),
                            (right_ear_x + ear_width, right_ear_y + ear_height), (0, 0, 255), 2)
                cv2.putText(frame, "PHONE?", (right_ear_x, right_ear_y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Check left ear region
        if left_ear_x >= 0 and left_ear_y + ear_height < frame.shape[0]:
            left_roi = mask[left_ear_y:left_ear_y + ear_height, 
                          left_ear_x:left_ear_x + ear_width]
            left_dark_pixels = cv2.countNonZero(left_roi) if left_roi.size > 0 else 0
            
            if left_dark_pixels > 200:  # Threshold for mobile detection
                mobile_detected = True
                cv2.rectangle(frame, (left_ear_x, left_ear_y),
                            (left_ear_x + ear_width, left_ear_y + ear_height), (0, 0, 255), 2)
                cv2.putText(frame, "PHONE?", (left_ear_x, left_ear_y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return mobile_detected

    def transform(self, frame):
        """Main processing function for each video frame"""
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.face_cascade is None:
            cv2.putText(img, "Error: OpenCV cascades not loaded", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        self.faces_detected = len(faces)
        ear_avg = 0.3
        mobile_detected = False
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Extract face region
            face_region = img[y:y + h, x:x + w]
            
            # Detect eyes and calculate EAR
            ear_values, eyes = self.detect_eyes_and_calculate_ear(face_region, img, x, y)
            self.eyes_detected = len(ear_values)
            
            # Calculate average EAR
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
                    
                    # Draw drowsiness alert
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.putText(img, "WAKE UP!", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Draw flashing rectangle around frame
                    if self.frame_count % 10 < 5:  # Flashing effect
                        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
            else:
                self.COUNTER = 0
                if self.ALARM_ON:
                    self.ALARM_ON = False
            
            # Mobile phone detection
            mobile_detected = self.detect_mobile_phone(img, x, y, w, h)
            
            if mobile_detected:
                self.MOBILE_COUNTER += 1
                if self.MOBILE_COUNTER >= self.MOBILE_THRESH:
                    if not self.MOBILE_ALERT:
                        self.MOBILE_ALERT = True
                    
                    cv2.putText(img, "MOBILE PHONE DETECTED!", (10, 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(img, "Please stop the car safely", (10, 140),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                self.MOBILE_COUNTER = max(0, self.MOBILE_COUNTER - 1)
                if self.MOBILE_COUNTER == 0:
                    self.MOBILE_ALERT = False
            
            # Display metrics on frame
            cv2.putText(img, f"EAR: {ear_avg:.3f}", (img.shape[1] - 200, img.shape[0] - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Eyes: {len(ear_values)}", (img.shape[1] - 200, img.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"Threshold: {self.EYE_AR_THRESH:.2f}", (img.shape[1] - 200, img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            break  # Process only the first face
        
        # Display frame counter
        cv2.putText(img, f"Frame: {self.frame_count}", (10, img.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    # Page configuration
    st.set_page_config(
        page_title="🚨 Driver Safety Monitor - Cloud Version",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
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
        <h1>🚨 Driver Safety Monitoring System</h1>
        <p>Live Drowsiness Detection + Mobile Phone Usage Alert</p>
        <small>✅ Cloud Compatible - Works on Streamlit Cloud</small>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = DrowsinessDetector()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        
        sensitivity = st.slider("👁️ Drowsiness Sensitivity", 0.1, 0.4, 0.25, 0.05)
        st.session_state.detector.EYE_AR_THRESH = sensitivity
        
        frame_threshold = st.slider("⏱️ Alert Frame Threshold", 10, 40, 20, 5)
        st.session_state.detector.EYE_AR_CONSEC_FRAMES = frame_threshold
        
        mobile_threshold = st.slider("📱 Mobile Detection Threshold", 5, 25, 15, 5)
        st.session_state.detector.MOBILE_THRESH = mobile_threshold
        
        st.markdown("### 📊 Instructions")
        st.info("""
        1. Click **START** to begin live monitoring
        2. Allow camera access when prompted
        3. Watch for alerts in the video and dashboard
        4. Adjust sensitivity as needed
        """)

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📹 Live Video Stream")
        
        # WebRTC streamer (works in Streamlit Cloud)
        webrtc_ctx = webrtc_streamer(
            key="driver-safety",
            video_transformer_factory=lambda: st.session_state.detector,
            client_settings=ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            ),
            async_processing=True,
        )

    with col2:
        st.markdown("### 📈 Live Dashboard")
        
        if webrtc_ctx.video_transformer:
            transformer = webrtc_ctx.video_transformer
            
            # Status cards
            if transformer.ALARM_ON:
                st.markdown('''
                <div class="alert-card">
                    <h3>😴 DROWSINESS ALERT!</h3>
                    <p>Driver appears drowsy - pull over safely!</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="status-good">
                    <h4>😊 Driver Alert</h4>
                </div>
                ''', unsafe_allow_html=True)
            
            if transformer.MOBILE_ALERT:
                st.markdown('''
                <div class="alert-card">
                    <h3>📱 MOBILE DETECTED!</h3>
                    <p>Please stop the car to use phone</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Live metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f'''
                <div class="metric-card">
                    <h4>Live EAR</h4>
                    <h2>{transformer.current_ear:.3f}</h2>
                    <small>Threshold: {transformer.EYE_AR_THRESH:.2f}</small>
                </div>
                ''', unsafe_allow_html=True)

            with col_m2:
                st.markdown(f'''
                <div class="metric-card">
                    <h4>Eyes Found</h4>
                    <h2>{transformer.eyes_detected}</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            col_m3, col_m4 = st.columns(2)
            with col_m3:
                st.markdown(f'''
                <div class="metric-card">
                    <h4>Drowsy Frames</h4>
                    <h2>{transformer.COUNTER}</h2>
                    <small>Max: {transformer.EYE_AR_CONSEC_FRAMES}</small>
                </div>
                ''', unsafe_allow_html=True)

            with col_m4:
                st.markdown(f'''
                <div class="metric-card">
                    <h4>Mobile Count</h4>
                    <h2>{transformer.MOBILE_COUNTER}</h2>
                    <small>Threshold: {transformer.MOBILE_THRESH}</small>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("👆 Start the video stream to see live metrics")

    # Information section
    with st.expander("ℹ️ How It Works - Cloud Version"):
        st.markdown('''
        ### ✅ Cloud Deployment Features
        - **WebRTC Technology**: Direct browser-to-app video streaming
        - **No Server Webcam**: Works on any hosting platform
        - **Real-time Processing**: Live video analysis
        - **Browser Compatible**: Works in Chrome, Firefox, Edge
        
        ### 🔧 Detection Algorithms
        1. **Face Detection**: OpenCV Haar Cascades
        2. **Eye Tracking**: Bilateral eye detection
        3. **EAR Calculation**: Eye Aspect Ratio monitoring
        4. **Mobile Detection**: Dark object detection near ears
        5. **Alert System**: Visual warnings and dashboard updates
        
        ### 🚀 Usage Tips
        - **Good Lighting**: Ensure face is well-lit
        - **Stable Position**: Keep steady for best results
        - **Camera Permission**: Allow browser camera access
        - **Adjust Sensitivity**: Use sliders to fine-tune
        
        ### ⚠️ Safety Notes
        - This is for demonstration/educational purposes
        - Real driving safety systems require professional implementation
        - Always prioritize actual road safety over technology testing
        ''')


if __name__ == "__main__":
    main()
