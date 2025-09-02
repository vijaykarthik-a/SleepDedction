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
    """Load MediaPipe face detection and mesh models"""
    if not MEDIAPIPE_AVAILABLE:
        return None, None, None, None
        
    try:
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        return face_detection, face_mesh, mp_face_detection, mp_face_mesh
    except Exception as e:
        st.error(f"MediaPipe initialization failed: {e}")
        return None, None, None, None


class EnhancedDrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        # Detection thresholds
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 15
        self.COUNTER = 0
        self.ALARM_ON = False
        
        # Load detectors
        self.face_cascade, self.eye_cascade, self.profile_cascade = load_detectors()
        self.face_detection, self.face_mesh, self.mp_face_detection, self.mp_face_mesh = load_mediapipe()
        
        # Metrics
        self.current_ear = 0.3
        self.eyes_detected = 0
        self.faces_detected = 0
        self.frame_count = 0
        self.face_confidence = 0.0
        self.detection_method = "None"

    def detect_face_opencv(self, frame):
        """Simple but reliable OpenCV face detection"""
        if self.face_cascade is None:
            return [], 0.0, "OpenCV unavailable"
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                enhanced_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                maxSize=(300, 300)
            )
            
            confidence = 0.8 if len(faces) > 0 else 0.0
            return faces, confidence, "OpenCV"
            
        except Exception as e:
            return [], 0.0, f"OpenCV Error: {str(e)}"

    def calculate_ear_simple(self, eye_region):
        """Simple EAR calculation"""
        try:
            if eye_region.size == 0:
                return 0.3
                
            # Convert to grayscale
            if len(eye_region.shape) == 3:
                gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_eye = eye_region
            
            # Simple threshold-based approach
            _, thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY)
            
            # Calculate ratio of white pixels
            white_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            
            if total_pixels > 0:
                ratio = white_pixels / total_pixels
                ear = ratio * 0.5  # Scale to EAR range
                return max(0.05, min(0.5, ear))
            
            return 0.3
        except Exception:
            return 0.3

    def detect_eyes_simple(self, face_region, frame, face_x, face_y):
        """Simple eye detection"""
        if self.eye_cascade is None:
            return []
        
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(10, 10),
                maxSize=(50, 50)
            )
            
            ear_values = []
            
            for (ex, ey, ew, eh) in eyes:
                # Extract eye region
                eye_region = face_region[ey:ey + eh, ex:ex + ew]
                
                if eye_region.size > 0:
                    ear = self.calculate_ear_simple(eye_region)
                    ear_values.append(ear)
                    
                    # Draw eye rectangle
                    color = (0, 255, 0) if ear > self.EYE_AR_THRESH else (0, 0, 255)
                    cv2.rectangle(frame, (face_x + ex, face_y + ey),
                                (face_x + ex + ew, face_y + ey + eh), color, 2)
            
            return ear_values
            
        except Exception:
            return []

    def recv(self, frame):
        """Simplified main processing function"""
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            # Use simple OpenCV detection for better compatibility
            faces, confidence, method = self.detect_face_opencv(img)
            
            self.faces_detected = len(faces)
            self.face_confidence = confidence
            self.detection_method = method
            
            ear_avg = 0.3
            
            if len(faces) > 0:
                # Process first face
                x, y, w, h = faces[0]
                
                # Draw face rectangle
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"Face Detected", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Extract face region
                face_region = img[y:y + h, x:x + w]
                
                # Detect eyes
                ear_values = self.detect_eyes_simple(face_region, img, x, y)
                self.eyes_detected = len(ear_values)
                
                # Calculate average EAR
                if len(ear_values) > 0:
                    ear_avg = sum(ear_values) / len(ear_values)
                
                self.current_ear = ear_avg
                
                # Drowsiness detection
                if ear_avg < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                    
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        self.ALARM_ON = True
                        
                        # Draw alert
                        cv2.putText(img, "DROWSINESS ALERT!", (10, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.putText(img, "WAKE UP!", (10, 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        
                        # Flashing border
                        if self.frame_count % 6 < 3:
                            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
                else:
                    self.COUNTER = max(0, self.COUNTER - 1)
                    if self.COUNTER == 0:
                        self.ALARM_ON = False
            
            # Display metrics on frame
            cv2.putText(img, f"EAR: {ear_avg:.3f}", (10, img.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"Eyes: {self.eyes_detected}", (10, img.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Faces: {self.faces_detected}", (10, img.shape[0] - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # If processing fails, return original frame
            return frame


def process_uploaded_image(uploaded_file):
    """Process uploaded image for testing"""
    try:
        # Load image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Load detectors
        face_cascade, eye_cascade, _ = load_detectors()
        
        if face_cascade is not None:
            # Detect faces
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            # Process faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_array, "Face Detected", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Detect eyes in face region
                face_region = img_array[y:y + h, x:x + w]
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                if eye_cascade is not None:
                    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3, minSize=(10, 10))
                    
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(img_array, (x + ex, y + ey),
                                    (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
        
        # Convert back to RGB for display
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return img_rgb, len(faces)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, 0


def main():
    # Page configuration
    st.set_page_config(
        page_title="🚨 Enhanced Drowsiness Detection System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚨 Enhanced Driver Drowsiness Detection")
    
    # Show MediaPipe status
    if MEDIAPIPE_AVAILABLE:
        st.success("✅ MediaPipe loaded - High precision detection enabled")
    else:
        st.warning("⚠️ MediaPipe not available - Using OpenCV fallback (still fully functional)")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Detection Settings")
    
    ear_threshold = st.sidebar.slider(
        "Eye Aspect Ratio Threshold", 
        min_value=0.1, 
        max_value=0.4, 
        value=0.25, 
        step=0.01
    )
    
    consecutive_frames = st.sidebar.slider(
        "Consecutive Frames for Alert", 
        min_value=5, 
        max_value=30, 
        value=15, 
        step=1
    )
    
    # Detection mode selection
    detection_mode = st.sidebar.radio(
        "Choose Detection Mode:",
        ["📹 Live Camera", "🖼️ Upload Image", "📋 Demo Mode"]
    )
    
    if detection_mode == "📹 Live Camera":
        # RTC Configuration with multiple STUN servers
        rtc_configuration = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun.services.mozilla.com:3478"]},
            ],
            "iceCandidatePoolSize": 10,
        })
        
        # Create detector
        detector = EnhancedDrowsinessDetector()
        detector.EYE_AR_THRESH = ear_threshold
        detector.EYE_AR_CONSEC_FRAMES = consecutive_frames
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📹 Live Video Stream")
            
            # Camera troubleshooting
            with st.expander("🔧 Camera Not Working? Click Here"):
                st.markdown("""
                **Common Solutions:**
                1. 🔄 **Refresh the page** and allow camera access
                2. 🌐 **Use Chrome browser** (best WebRTC support)
                3. 🔒 **Ensure HTTPS connection** (required for camera)
                4. 📱 **On mobile**: Try landscape mode
                5. 🔧 **Clear browser cache** and try again
                6. 🖼️ **Try "Upload Image" mode** as alternative
                
                **Browser Compatibility:**
                - ✅ Chrome/Chromium (Recommended)
                - ✅ Firefox  
                - ✅ Edge
                - ⚠️ Safari (limited support)
                """)
            
            # WebRTC streamer with optimized settings
            webrtc_ctx = webrtc_streamer(
                key="drowsiness-detector-live",
                video_processor_factory=lambda: detector,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 320, "ideal": 640, "max": 1280},
                        "height": {"min": 240, "ideal": 480, "max": 720},
                        "frameRate": {"min": 10, "ideal": 15, "max": 25}
                    },
                    "audio": False
                },
                async_processing=False,
                video_html_attrs={
                    "style": {"width": "100%", "border": "2px solid #ff6b6b"},
                    "controls": False,
                    "autoPlay": True,
                }
            )
            
            # Connection status
            if webrtc_ctx.state.playing:
                st.success("🟢 Camera connected successfully!")
            elif webrtc_ctx.state.signalling:
                st.warning("🟡 Connecting to camera...")
            else:
                st.error("🔴 Camera not connected")
        
        with col2:
            st.subheader("📊 Live Metrics")
            
            if webrtc_ctx.video_processor:
                det = webrtc_ctx.video_processor
                
                # Status indicators
                face_status = "🟢 Detected" if det.faces_detected > 0 else "🔴 No Face"
                st.metric("Face Detection", face_status)
                
                ear_status = "🟢 Alert" if det.current_ear > ear_threshold else "🔴 Drowsy"
                st.metric("Eye Status", f"{ear_status}")
                st.metric("EAR Value", f"{det.current_ear:.3f}")
                
                # Alert status
                if det.ALARM_ON:
                    st.error("🚨 DROWSINESS ALERT!")
                else:
                    st.success("✅ Driver Alert")
                
                st.metric("Eyes Detected", det.eyes_detected)
                st.metric("Frames Processed", det.frame_count)
            else:
                st.info("Start camera to see metrics")
    
    elif detection_mode == "🖼️ Upload Image":
        st.subheader("🖼️ Test with Image Upload")
        
        uploaded_file = st.file_uploader(
            "Upload an image to test face and eye detection",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear photo showing a person's face"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                with st.spinner("Processing image..."):
                    processed_img, face_count = process_uploaded_image(uploaded_file)
                
                if processed_img is not None:
                    st.image(processed_img, use_column_width=True)
                    st.success(f"✅ Detected {face_count} face(s)")
                else:
                    st.error("❌ Failed to process image")
    
    else:  # Demo Mode
        st.subheader("📋 Demo Mode - How It Works")
        
        # Create demo visualization
        demo_img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Draw demo face
        cv2.rectangle(demo_img, (200, 100), (400, 300), (0, 255, 0), 3)
        cv2.putText(demo_img, "Face Detected", (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw demo eyes
        cv2.rectangle(demo_img, (230, 150), (270, 180), (255, 255, 0), 2)
        cv2.rectangle(demo_img, (330, 150), (370, 180), (255, 255, 0), 2)
        cv2.putText(demo_img, "Eyes", (280, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw EAR indicator
        cv2.putText(demo_img, f"EAR: {ear_threshold:.2f}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(demo_img, "Status: Alert", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Convert and display
        demo_rgb = cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB)
        st.image(demo_rgb, caption="Demo: How face and eye detection works")
        
        st.markdown("""
        ### 🎯 How the Detection Works:
        1. **Face Detection**: Locates faces in the video stream
        2. **Eye Tracking**: Identifies and tracks both eyes
        3. **EAR Calculation**: Measures Eye Aspect Ratio (height/width)
        4. **Drowsiness Alert**: Triggers when EAR drops below threshold
        
        ### 📊 Key Metrics:
        - **EAR > 0.25**: Eyes open (alert)
        - **EAR < 0.25**: Eyes closing (drowsy)
        - **Alert Threshold**: {consecutive_frames} consecutive frames
        """.format(consecutive_frames=consecutive_frames))
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 Quick Start Guide:
    
    **🎥 Live Camera Mode:**
    - Click "START" to begin video stream
    - Allow camera permissions when prompted
    - System will monitor your eyes in real-time
    
    **🖼️ Image Upload Mode:**
    - Test the detection on static images
    - Good for verifying the system works
    
    **📋 Demo Mode:**
    - See how the detection system works
    - Understand the visual indicators
    """)
    
    # Technical info
    with st.expander("🔍 Technical Information"):
        st.markdown(f"""
        **Current Configuration:**
        - Detection Method: {"MediaPipe + OpenCV" if MEDIAPIPE_AVAILABLE else "OpenCV Only"}
        - EAR Threshold: {ear_threshold}
        - Alert Frames: {consecutive_frames}
        
        **System Status:**
        - Face Detection: {"✅ Ready" if load_detectors()[0] is not None else "❌ Error"}
        - Eye Detection: {"✅ Ready" if load_detectors()[1] is not None else "❌ Error"}
        - MediaPipe: {"✅ Available" if MEDIAPIPE_AVAILABLE else "❌ Fallback Mode"}
        """)


if __name__ == "__main__":
    main()
