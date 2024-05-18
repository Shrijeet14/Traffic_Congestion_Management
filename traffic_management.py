import pyrebase
import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import time 
from datetime import datetime
import base64
from inference_sdk import InferenceHTTPClient
import os
import uuid

# Create a temporary directory for saving frames
TEMP_FRAMES_DIR = 'temp_frames'
os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)


# *************************************************FIREBASE*****************************************************

# Configuration Key
firebaseConfig = {
  'apiKey': "AIzaSyCFLtp3C57X5WXzMDtUSyftcaaTa7S-CBE",
  'authDomain': "datathon-4031d.firebaseapp.com",
  'projectId': "datathon-4031d",
  'databaseURL':'https://datathon-4031d-default-rtdb.europe-west1.firebasedatabase.app/',
  'storageBucket': "datathon-4031d.appspot.com",
  'messagingSenderId': "662523371946",
  'appId': "1:662523371946:web:d9deb57fc8ed8761238bd3",
  'measurementId': "G-WES81XPRF3"
}

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database 
db = firebase.database()
storage = firebase.storage()

# **************************************************************************************************************











# *************************************Traffic flow control functions*******************************************

# Traffic flow control functions
present_time = str(time.ctime(time.time()))
# Function to generate video frames
def get_video_frames_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

class Detections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id

def draw_shapes(frame):
    shapes = []
    clone = frame.copy()
    num_points_selected = 0
    points = []

    def draw(event, x, y, flags, param):
        nonlocal num_points_selected, points

        if event == cv2.EVENT_LBUTTONDOWN:
            if num_points_selected == 0:
                points = [(x, y)]
                num_points_selected += 1
            elif num_points_selected < 4:
                cv2.line(clone, points[-1], (x, y), (0, 0, 255), 2)
                points.append((x, y))
                num_points_selected += 1
            elif num_points_selected == 4:
                shapes.append(points)
                cv2.polylines(clone, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
                num_points_selected = 0
                points = []

    cv2.namedWindow("Draw Shapes (Press 'q' to quit)")
    cv2.setMouseCallback("Draw Shapes (Press 'q' to quit)", draw)

    while True:
        cv2.imshow("Draw Shapes (Press 'q' to quit)", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return shapes
# **************************************************************************************************************










# *************************************Congestion flow control functions****************************************
# Congestion control functions
class Detections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id

def box_annotator(frame, detections, labels):
    for i in range(len(detections.xyxy)):
        xyxy = detections.xyxy[i]
        class_id = detections.class_id[i]
        confidence = detections.confidence[i]
        label = labels[i]

        x1, y1, x2, y2 = xyxy.astype(int)
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)

    return frame

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="REKRi3n8N6cMWAJxLl4e"
)

def annotate_frame(frame, detections):
    for detection in detections:
        x1, y1, width, height = detection["x"], detection["y"], detection["width"], detection["height"]
        label = detection["class"]
        confidence = detection["confidence"]

        x2 = x1 + (width/2)
        y2 = y1 - (height/2)

        x1 = x1 - (width/2)
        y1 = y1 + (height/2)
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
# **************************************************************************************************************









# ****************************************************UI********************************************************

st.sidebar.title("Traffic Congestion Detection Webapp")

# Initialize session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Authentication
login_signup_container = st.sidebar.empty()

if not st.session_state.logged_in:
    with login_signup_container:
        choice = st.sidebar.radio('Login/Signup', ['Login','Sign up'])
        email = st.sidebar.text_input('Please enter your email address')
        password = st.sidebar.text_input('Please enter your password', type='password')

    if choice == 'Sign up':
        handle = st.sidebar.text_input('Please input your app handle name', value='Default')
        submit = st.sidebar.button('Create my account')

        if submit:
            user = auth.create_user_with_email_and_password(email, password)
            st.success('Your account is created successfully')
            user = auth.sign_in_with_email_and_password(email, password)
            db.child(user['localId']).child("Handle").set(handle)
            db.child(user['localId']).child("Id").set(user['localId'])
            st.title('Welcome ' + handle)
            st.info('Login via login drop down selection')

    if choice == 'Login':
        login = st.sidebar.checkbox('Login')
        if login:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state.logged_in = True
            st.session_state.user = user
            st.session_state.handle_ref = db.child(user['localId']).child("Handle").get().val()
            st.experimental_rerun()

if st.session_state.logged_in:
    with login_signup_container:
        login_signup_container.empty()
        logout_clicked = st.sidebar.button('Logout')

    if logout_clicked:
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.handle_ref = None
        st.experimental_rerun()

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    st.title(st.session_state.handle_ref)
    st.sidebar.title('MODELS')
    options =st.sidebar.radio('SELECT : ',['Congestion Detector Model' , 'Lane Congestion Detector Model' , 'Pothole Detector Model'])

    
    if options == 'Lane Congestion Detector Model':
                    
        st.title("Vehicle and Person Detection Web App")

        col1 , col2 = st.columns(2)

        uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

        graph_container = st.empty()

        frame_counter=0

        # Live Message and button for Firebase
        st.sidebar.title("Live Message for Client")
        live_data = st.sidebar.text_area("Type Your Message :-")
        if st.sidebar.button("Send Message To Client"):
            local_id = st.session_state.user['localId']
            timestamp = datetime.now().strftime("%Y_%m_%d_at_%H-%M-%S")
            db.child(local_id).child("Live_Message").child(timestamp).set(live_data)
            # Define the custom HTML and CSS
            html_code = '''
            <div style="color: #F11181; background-color: #64A0E6; padding: 10px; border-radius: 5px;">
                Data sent to Firebase!
            </div>
            '''
            # Display the custom message in the sidebar
            st.sidebar.markdown(html_code, unsafe_allow_html=True)

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            # st.sidebar.markdown("## Select Shapes")
            with col1 :
                video_player = st.video(video_path)
            with col2 :
                updated_frame_container = st.empty()

            # Load the first frame to select shapes
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                # first_frame_container.image(frame, channels="BGR", caption="Draw shapes on the first frame")
                shapes = draw_shapes(frame)

                if shapes:
                    # Initialize YOLO model
                    model = YOLO("yolov8x.pt")
                    model.fuse()

                    # Get class names dictionary
                    CLASS_NAMES_DICT = model.model.names

                    # Class IDs of interest
                    VEHICLE_CLASS_ID = [2, 3, 5, 7]
                    PERSON_CLASS_ID = [0]

                    # Create frame generator
                    generator = get_video_frames_generator(video_path)

                    # Initialize Plotly figure
                    fig = make_subplots(rows=len(shapes), cols=1, subplot_titles=[f"Lane {i+1}" for i in range(len(shapes))])

                    # Process video frames
                    vehicle_counts = defaultdict(list)
                    for frame in generator:
                        # Model prediction on single frame and conversion to supervision Detections
                        frame_counter += 1

                        if frame_counter % 10 != 0:
                            continue

                        results = model(frame)
                        detections = results[0].boxes
                        detections = Detections(
                            xyxy=detections.xyxy.cpu().numpy(),
                            confidence=detections.conf.cpu().numpy(),
                            class_id=detections.cls.cpu().numpy().astype(int)
                        )

                        # Count objects within selected shapes
                        for idx, shape in enumerate(shapes, start=1):
                            vehicle_count = 0
                            for xyxy, class_id in zip(detections.xyxy, detections.class_id):
                                polygon = np.array(shape)
                                if cv2.pointPolygonTest(polygon, (xyxy[0], xyxy[1]), False) >= 0:
                                    if class_id in VEHICLE_CLASS_ID:
                                        vehicle_count += 1
                            vehicle_counts[idx].append(vehicle_count)

                        # Annotate frame
                        for shape in shapes:
                            cv2.polylines(frame, [np.array(shape)], isClosed=True, color=(0, 255, 0), thickness=2)

                        # Display the frame in Streamlit
                        updated_frame_container.image(frame, channels="BGR", use_column_width=True)

                        # Update Plotly figure
                        for idx, shape_count in vehicle_counts.items():
                            fig.add_trace(go.Scatter(y=shape_count, mode='lines', name=f'LANE_{idx}'), row=idx, col=1)

                        # Update plot
                        graph_container.write(fig)
                # Initialize lane-wise vehicle count dictionary
                lane_vehicle_counts = defaultdict(int)

                # Populate lane_vehicle_counts with vehicle counts for each lane
                for idx, shape_count_list in vehicle_counts.items():
                    lane_vehicle_counts[f"LANE_{idx}"] = sum(shape_count_list)

                # Display lane-wise vehicle counts in a Streamlit table
                st.write("Lane-wise Vehicle Counts")
                st.table(lane_vehicle_counts)
        
        # Data box and button for Firebase
            st.sidebar.title("Data Box")
            mssg_data = st.sidebar.text_area("Lane Data" , value=str(lane_vehicle_counts))
            if st.sidebar.button("Send Data to Clients"):
                local_id = st.session_state.user['localId']
                timestamp = datetime.now().strftime("%Y_%m_%d_at_%H-%M-%S")
                db.child(local_id).child("Lane_data").child(timestamp).set(mssg_data)
                # st.sidebar.success("Data sent to Firebase!")
                # Define the custom HTML and CSS
                html_code = '''
                <div style="color: #F11181; background-color: #64A0E6; padding: 10px; border-radius: 5px;">
                    Data sent to Firebase!
                </div>
                '''
                # Display the custom message in the sidebar
                st.sidebar.markdown(html_code, unsafe_allow_html=True)



    if options == "Congestion Detector Model":
        # Congestion detection variables
        congestion_threshold = 3
        congestion_frame_threshold = 2
        congestion_counter = defaultdict(int)
        congestion_frames_counter = defaultdict(int)
        congestion_saved_counter = defaultdict(int)

        # Initialize trackers for vehicles and persons
        vehicle_trackers = defaultdict(list)
        person_trackers = defaultdict(list)

        # Initialize previous detections for vehicles and persons
        prev_vehicle_detections = defaultdict(list)
        prev_person_detections = defaultdict(list)

        # # Define source video path and model
        # SOURCE_VIDEO_PATH = "D:\\HACKATHONS\\Datathon\\try4\\test_video1.mp4"
        MODEL = "yolov8x.pt"

        # Initialize Streamlit app
        st.title("Congestion Detection Model")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
        graph_container = st.empty()

        # Initialize congestion table
        congestion_table = st.empty()


        # Live Message and button for Firebase
        st.sidebar.title("Live Message for Client")
        live_data = st.sidebar.text_area("Type Your Message :-")
        if st.sidebar.button("Send Message To Client"):
            local_id = st.session_state.user['localId']
            timestamp = datetime.now().strftime("%Y_%m_%d_at_%H-%M-%S")
            db.child(local_id).child("Live_Message").child(timestamp).set(live_data)
            # Define the custom HTML and CSS
            html_code = '''
            <div style="color: #F11181; background-color: #64A0E6; padding: 10px; border-radius: 5px;">
                Data sent to Firebase!
            </div>
            '''
            # Display the custom message in the sidebar
            st.sidebar.markdown(html_code, unsafe_allow_html=True)


        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                # video_path = tmp_file.name
                SOURCE_VIDEO_PATH = tmp_file.name

            # Initialize YOLO model
            model = YOLO(MODEL)
            model.fuse()

            # Get class names dictionary
            CLASS_NAMES_DICT = model.model.names

            # Class IDs of interest
            VEHICLE_CLASS_ID = [2, 3, 5, 7]
            PERSON_CLASS_ID = [0]

            # Create frame generator
            generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

            # Initialize congestion messages list
            congestion_data = []

            # Process video frames
            frame_count = 0
            for frame in generator:
                frame_count += 1
                if frame_count % 20 == 0:
                    # Model prediction on single frame and conversion to supervision Detections
                    results = model(frame)
                    detections = results[0].boxes
                    detections = Detections(
                        xyxy=detections.xyxy.cpu().numpy(),
                        confidence=detections.conf.cpu().numpy(),
                        class_id=detections.cls.cpu().numpy().astype(int)
                    )

                    # Format custom labels
                    labels = [
                        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                        for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id)
                    ]

                    # Annotate frame
                    frame = box_annotator(frame=frame, detections=detections, labels=labels)

                    # Congestion detection
                    for class_id, xyxy in zip(detections.class_id, detections.xyxy):
                        # Check if class is vehicle or person
                        if class_id in VEHICLE_CLASS_ID or class_id in PERSON_CLASS_ID:
                            # Convert xyxy to centroid
                            centroid_x = (xyxy[0] + xyxy[2]) / 2
                            centroid_y = (xyxy[1] + xyxy[3]) / 2

                            # Update congestion counter
                            congestion_counter[(class_id, centroid_x, centroid_y)] += 1

                            # Check if congestion threshold is reached
                            if congestion_counter[(class_id, centroid_x, centroid_y)] >= congestion_threshold:
                                congestion_frames_counter[(class_id, centroid_x, centroid_y)] += 1

                                # Check if congestion frame threshold is reached
                                if congestion_frames_counter[(class_id, centroid_x, centroid_y)] >= congestion_frame_threshold:
                                    # Check if congestion has already been saved for this entity three times
                                    if congestion_saved_counter[(class_id, centroid_x, centroid_y)] < 3:
                                        # Mark congestion with circle
                                        farthest_distance = 0
                                        for other_class_id, other_xyxy in zip(detections.class_id, detections.xyxy):
                                            if class_id != other_class_id:
                                                other_centroid_x = (other_xyxy[0] + other_xyxy[2]) / 2
                                                other_centroid_y = (other_xyxy[1] + other_xyxy[3]) / 2
                                                distance = np.sqrt((centroid_x - other_centroid_x)**2 + (centroid_y - other_centroid_y)**2)
                                                farthest_distance = max(farthest_distance, distance)

                                        # Mark congestion with yellow circle
                                        color = (0, 255, 255)  # Yellow color for congestion circle
                                        thickness = 2
                                        cv2.circle(frame, (int(centroid_x), int(centroid_y)), int(farthest_distance / 5), color, thickness)

                                        # Draw red lines connecting centers of all rectangular frames inside the congestion circle
                                        for other_class_id, other_xyxy in zip(detections.class_id, detections.xyxy):
                                            if class_id != other_class_id:
                                                other_centroid_x = (other_xyxy[0] + other_xyxy[2]) / 2
                                                other_centroid_y = (other_xyxy[1] + other_xyxy[3]) / 2
                                                distance = np.sqrt((centroid_x - other_centroid_x)**2 + (centroid_y - other_centroid_y)**2)
                                                if distance <= (farthest_distance / 5):
                                                    cv2.line(frame, (int(centroid_x), int(centroid_y)), (int(other_centroid_x), int(other_centroid_y)), (0, 0, 255), thickness)

                                        # Append congestion message and frame
                                        timestamp = datetime.now().strftime("%Y-%m-%d_at_%H-%M-%S")
                                        frame_filename = f"{timestamp}_{uuid.uuid4()}.jpg"
                                        congestion_frame_path = os.path.join(TEMP_FRAMES_DIR, frame_filename)

                                        # Save congestion frame
                                        cv2.imwrite(congestion_frame_path, frame)

                                        # Upload to Firebase Storage
                                        storage.child(f"congestion_frames/{frame_filename}").put(congestion_frame_path)
                                        frame_url = storage.child(f"congestion_frames/{frame_filename}").get_url(None)


                                        congestion_message = f"Congestion detected at {timestamp}"
                                        congestion_data.append((congestion_message, frame_url))

                                        # Update congestion table with icon-sized frames
                                        congestion_table.image([frame_url for _, frame_url in congestion_data], width=100)

                                        # Update congestion table
                                        congestion_table.table(congestion_data)

                                        # Increment congestion saved counter
                                        congestion_saved_counter[(class_id, centroid_x, centroid_y)] += 1

                                    # Reset congestion frames counter
                                    congestion_frames_counter[(class_id, centroid_x, centroid_y)] = 0

                    # Display the frame in a window
                    cv2.imshow('Frame', frame)

                    # Set a small delay between frames for smoother playback
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            # Data box and button for Firebase
            st.sidebar.title("Data Box")
            mssg_data = st.sidebar.text_area("Congestion Data" , value=str(congestion_data))
            if st.sidebar.button("Send Data to Firebase"):
                local_id = st.session_state.user['localId']
                timestamp = datetime.now().strftime("%Y_%m_%d_at_%H-%M-%S")
                db.child(local_id).child("congestion_data").child(timestamp).set(mssg_data)
                # st.sidebar.success("Data sent to Firebase!")
                # Define the custom HTML and CSS
                html_code = '''
                <div style="color: #F11181; background-color: #64A0E6; padding: 10px; border-radius: 5px;">
                    Data sent to Firebase!
                </div>
                '''
                # Display the custom message in the sidebar
                st.sidebar.markdown(html_code, unsafe_allow_html=True)

                


    if options =='Pothole Detector Model':
        # UI
        st.title("Pothole detection Model")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
        road_name = st.text_input("Enter the name of the road:")

        table_container =st.empty()

        # Live Message and button for Firebase
        st.sidebar.title("Live Message for Client")
        live_data = st.sidebar.text_area("Type Your Message :-")
        if st.sidebar.button("Send Message To Client"):
            local_id = st.session_state.user['localId']
            timestamp = datetime.now().strftime("%Y_%m_%d_at_%H-%M-%S")
            db.child(local_id).child("Live_Message").child(timestamp).set(live_data)
            # Define the custom HTML and CSS
            html_code = '''
            <div style="color: #F11181; background-color: #64A0E6; padding: 10px; border-radius: 5px;">
                Data sent to Firebase!
            </div>
            '''
            # Display the custom message in the sidebar
            st.sidebar.markdown(html_code, unsafe_allow_html=True)




        if uploaded_file is not None and road_name:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name


            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error opening video file")
                st.stop()


            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Define codec and create VideoWriter object
            out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            pothole_count = 0

            table_container.table({"Road Name": [road_name], "Pothole Count": [pothole_count]})
            frame_counter =  0

            detected_points = []  # List to maintain detected points

            RADIUS = 100

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                frame_counter += 1

                if frame_counter % 10 != 0:
                    continue

                # Send the frame to Roboflow API for prediction
                response = CLIENT.infer(frame_base64, model_id="pothole-detection-gh7b3/1")

                # Extract detections
                detections = response["predictions"]

                # Filter redundant detections
                filtered_detections = []
                for detection in detections:
                    x, y = detection["x"], detection["y"]
                    label = detection["class"]
                    confidence = detection["confidence"]
                    
                    # Calculate center point of the bounding box
                    center_x = int(x)
                    center_y = int(y)
                    
                    # Check if the point is within a radius from any detected point
                    redundant = False
                    for point in detected_points:
                        if np.linalg.norm(np.array([center_x, center_y]) - np.array(point)) < RADIUS:
                            redundant = True
                            break
                    
                    if not redundant:
                        # Add point to detected points list
                        detected_points.append([center_x, center_y])
                        filtered_detections.append(detection)

                # Annotate frame with filtered detections
                annotated_frame = annotate_frame(frame, filtered_detections)

                # Count potholes
                pothole_count += len(filtered_detections)

                # Write the annotated frame to the output video
                out.write(annotated_frame)

                # for point in detected_points:
                #     cv2.circle(frame, (point[0], point[1]), RADIUS, (255, 0, 0), 2)

                table_container.table({"Road Name": [road_name], "Pothole Count": [pothole_count]})
                # Draw circles around detected points with the range considered
                

                # Display the frame (optional)
                cv2.imshow('Annotated Video', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release video capture and writer objects
            cap.release()
            out.release()

            # Display pothole count
            mssg =f"Potholes detected on {road_name}: {pothole_count}"
            st.success(mssg)


        # Data box and button for Firebase
            st.sidebar.title("Data Box")
            mssg_data = st.sidebar.text_area("Pothole Data" , value=mssg)
            if st.sidebar.button("Send Data to Clients"):
                local_id = st.session_state.user['localId']
                timestamp = datetime.now().strftime("%Y_%m_%d_at_%H-%M-%S")
                db.child(local_id).child("Pothole_data").child(timestamp).set(mssg_data)
                # st.sidebar.success("Data sent to Firebase!")
                # Define the custom HTML and CSS
                html_code = '''
                <div style="color: #F11181; background-color: #64A0E6; padding: 10px; border-radius: 5px;">
                    Data sent to Firebase!
                </div>
                '''
                # Display the custom message in the sidebar
                st.sidebar.markdown(html_code, unsafe_allow_html=True)

else:
    st.sidebar.write("Please log in to use the app.")

