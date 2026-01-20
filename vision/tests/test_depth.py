import cv2
import numpy as np

# StereoBM matcher (can also use StereoSGBM for better results)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
# stereo = cv2.StereoSGBM(numDisparities=64, blockSize=15)

cap_left = cv2.VideoCapture("sphere_left.mp4")
cap_right = cv2.VideoCapture("sphere_right.mp4")

# Get frame width and height
# image_width_px = 3280
# image_height_px = 2464

# Camera parameters (from calibration/spec sheet)
f_mm = 2.6  # focal length in mm
sensor_width_mm = 3.6  # sensor width in mm
sensor_height_mm = 2.7  # sensor height in mm 
baseline_m = 60/1000
image_width_px  = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height_px = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Convert focal length to pixels
fx = (f_mm * image_width_px) / sensor_width_mm
fy = (f_mm * image_height_px) / sensor_height_mm


fps = cap_left.get(cv2.CAP_PROP_FPS) # Get the frame rate of the video
delay = int(1000 / fps)  # Delay between frames in milliseconds


while cap_left.isOpened():
    # Load stereo images
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    cv2.imwrite('img_left.jpg',frame_left)
    cv2.imwrite('img_right.jpg',frame_right)

    # Grayscale conversion
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Compute disparity (it will be a 16-bit signed image, scaled by 16)
    disparity = stereo.compute(gray_left, gray_right)
    # Normalize for display
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # Convert to float32 and avoid division by zero
    disparity = disparity.astype(np.float32) / 16.0
    disparity[disparity == 0] = 0.1  # to prevent division by zero

    # Depth map (Z in meters)
    depth_map = (fx * baseline_m) / disparity

    # Display the frame
    cv2.imshow("Left frame", frame_left)
    cv2.imshow("Right frame", frame_right)
    cv2.imshow("Disparity", disp_vis)
    cv2.imshow("Depth Map", cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    # cv2.waitKey(0)


    # Press 'q' to quit
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap_left.release()
cap_right.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
