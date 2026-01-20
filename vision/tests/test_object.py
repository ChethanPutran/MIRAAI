import cv2
import cv2
# from extractor import Extractor
from vision.estimator.object_identifier import ObjectIdentifier
# from deep_sort_realtime.deepsort_tracker import DeepSort


from vision.estimator.object_identifier import ObjectIdentifier

oi = ObjectIdentifier()
model = oi.get_model()

# Initialize Deep SORT tracker
# tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)  # You can tweak parameters
tracker_l = cv2.TrackerCSRT_create()  # You can also try TrackerKCF_create()
tracker_r = cv2.TrackerCSRT_create()  # You can also try TrackerKCF_create()


# Stereo camera parameters (example values — replace with your calibration data)
f = 2.6  # in mm
b = 60   # in mm

fx = 718.856  # focal length in pixels
fy = 718.856
cx = 607.1928
cy = 185.2157
baseline = 60/1000  # in meters

# Load the stereo capture
cap_left = cv2.VideoCapture("sphere_left.mp4")
cap_right = cv2.VideoCapture("sphere_right.mp4")




object_ = None
object_2 = None
object_initialized = False

class Object:
    def __init__(self,name,x1,y1,x2,y2,conf):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.center = ( (x1+x2)//2, (y1+y2)//2 )

    def track(self):
        pass
    def update(self):
        pass

def get_object_props(result,first_only=True):
    objects = []
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        if first_only:
            return Object(class_id,*cords,conf)
        objects.append(Object(class_id,*cords,conf))
    return objects
    

while cap_left.isOpened():
    # Load stereo images
    # left_img = cv2.imread('left.png')
    # right_img = cv2.imread('right.png')
    ret1, left_img = cap_left.read()
    ret2, right_img = cap_right.read()

    if (not ret1) or (not ret2):
        break

    if not object_initialized:

        # Run YOLOv5 on both images
        model.conf = 0.4  # confidence threshold
        results_left = model(left_img)
        results_right = model(right_img)

        # Parse YOLO detections
        # objects_left = get_object_props(results_left[0],first_only=False)
        # objects_right = get_object_props(results_right[0],first_only=False)

        objects_3d = []

        # Match and triangulate
        for object_l in objects_left:
            class_id_L = object_l['class_id']
            cxL = object_l
            cyL = (boxL[1] + boxL[3]) / 2

            # Find best match in right image (same class, closest y)
            min_y_diff = float('inf')
            matched_boxR = None
            for boxR in obj:
                class_id_R = int(boxR[5])
                if class_id_R != class_id_L:
                    continue
                cyR = (boxR[1] + boxR[3]) / 2
                if abs(cyL - cyR) < min_y_diff:
                    min_y_diff = abs(cyL - cyR)
                    matched_boxR = boxR

            if matched_boxR is not None:
                cxR = (matched_boxR[0] + matched_boxR[2]) / 2

                # Disparity
                disparity = cxL - cxR
                if disparity == 0:
                    continue

                # Depth (Z)
                Z = (fx * baseline) / disparity
                X = (cxL - cx) * Z / fx
                Y = (cyL - cy) * Z / fy

                # Estimate 3D box size
                width_px = boxL[2] - boxL[0]
                height_px = boxL[3] - boxL[1]
                width_m = (width_px * Z) / fx
                height_m = (height_px * Z) / fy
                depth_m = 0.2  # Approximate or infer from stereo, or use known object depth

                objects_3d.append({
                    'class_name': names[class_id_L],
                    'position_3d': (round(X, 3), round(Y, 3), round(Z, 3)),
                    'size_3d': (round(width_m, 3), round(height_m, 3), round(depth_m, 3)),
                })

        tracker_l.init(left_img, boxL)
        tracker_r.init(right_img, boxR)
        # Print results
        for obj in objects_3d:
            print(f"[{obj['class_name']}] → 3D Pos: {obj['position_3d']}, Size: {obj['size_3d']}")

    else:
        success_l, bbox_l = tracker_l.update(left_img)
        success_r, bbox_r = tracker_r.update(right_img)

        if success_l:
            # Draw tracked box
            x, y, w, h = [int(i) for i in bbox_l]
            cv2.rectangle(left_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(left_img, "Tracking Left", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        else:
            cv2.putText(left_img, "Lost Left", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
        if success_r:
            # Draw tracked box
            x, y, w, h = [int(i) for i in bbox_r]
            cv2.rectangle(left_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(left_img, "Tracking Left", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        else:
            cv2.putText(left_img, "Lost Right", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
       

    # Display the frame
    cv2.imshow("Detected Object Left", left_img)
    cv2.imshow("Detected Object Right", right_img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break




cap_left = cv2.VideoCapture("sphere_left.mp4")
cap_right = cv2.VideoCapture("sphere_right.mp4")
fps = cap_left.get(cv2.CAP_PROP_FPS) # Get the frame rate of the video
delay = int(1000 / fps)  # Delay between frames in milliseconds

oi = ObjectIdentifier()
model = oi.get_model()



while cap_left.isOpened():
    # Load stereo images
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if (not ret1) or (not ret2):
        break

    # Run detection and tracking using YOLO
    # results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Run YOLO detection on both
    if not object_initialized:
        results_l = oi.identify_objects(frame_left)
        results_r = oi.identify_objects(frame_right)

        result_l = results_l[0]
        result_r = results_r[0]

        #detections = []
        
        obj_l = get_object_props(result_l)
        obj_r = get_object_props(result_r)

        


        # Matche the same object between left and right

        # Computes disparity and depth

        # Estimates 3D position and bounding box dimensions

        (x1_l, y1_l), (x2_l, y2_l) = obj_l['cords']
        (x1_r, y1_r), (x2_r, y2_r) = obj_r['cords']

        cx_left = (x1_l + x2_l) // 2
        cx_right = (x1_r + x2_r) // 2

        disparity = cx_left - cx_right
        fx=fy=fz = f = 10
        baseline = 10
        cx = 1
        cy = 1

        Z = (fx * baseline) / disparity  # Depth in meters
        X = (cx_left - cx) * Z / fx
        Y = (cy_left - cy) * Z / fy

        bbox = (x1, y1, x2 - x1, y2 - y1)
        # Initialize tracker with the first frame and bounding box
        tracker.init(frame, bbox)
        object_initialized = True
    
    else:
        success, bbox = tracker.update(frame)

        if success:
            # Draw tracked box
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        # Update Deep SORT tracker
        #tracks = tracker.update_tracks(detections, frame=frame)

        # Draw tracking results
        # for track in tracks:
        #     if not track.is_confirmed():
        #         continue

        #     track_id = track.track_id
        #     l, t, w, h = track.to_ltrb()
        #     x1, y1, x2, y2 = int(l), int(t), int(l + w), int(t + h)

    
        # print(results[0])
        # Visualize results
        # annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow("Detected Object", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

    