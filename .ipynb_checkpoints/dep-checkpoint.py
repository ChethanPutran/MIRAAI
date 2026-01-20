import cv2
import numpy as np

MATCH_POINTS = 10
def depth_mapper():
    # StereoBM matcher (can also use StereoSGBM for better results)
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    # stereo = cv2.StereoSGBM(numDisparities=64, blockSize=15)

    frame_left = cv2.imread("img_left.jpg")
    frame_right = cv2.imread("img_right.jpg")

    # Get frame width and height
    # image_width_px = 3280
    # image_height_px = 2464

    # Camera parameters (from calibration/spec sheet)
    f_mm = 2.6  # focal length in mm
    sensor_width_mm = 3.6  # sensor width in mm
    sensor_height_mm = 2.7  # sensor height in mm 
    baseline_m = 60/1000
    image_width_px  = frame_left.shape[1]
    image_height_px =frame_left.shape[0]

    # Convert focal length to pixels
    fx = (f_mm * image_width_px) / sensor_width_mm
    fy = (f_mm * image_height_px) / sensor_height_mm


    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Compute disparity (it will be a 16-bit signed image, scaled by 16)
    disparity = stereo.compute(gray_left, gray_right)
    # Normalize for display
    # disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disparity)

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
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def feature_match(imgL,imgR):
    # 1. Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # 2. Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)

    # 3. Match using BFMatcher (Hamming for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 4. Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. Draw top N matches
    img_matches = cv2.drawMatches(imgL, kp1, imgR, kp2, matches[:MATCH_POINTS], None, flags=2)
    
    # Add index labels on top of matched points
    avg_desparity = 0

    points = []
    for i, match in enumerate(matches[:MATCH_POINTS]):
        pt_left = tuple(map(int, kp1[match.queryIdx].pt))
        pt_right = tuple(map(int, kp2[match.trainIdx].pt))
        points.append((pt_left,pt_right))
        disparity = pt_left[0] - pt_right[0]  # x_left - x_right

        avg_desparity+=disparity

        # Offset x for the right image in combined drawMatches output
        offset = imgL.shape[1]  # width of left image

        cv2.putText(img_matches, str(i), pt_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        pt_right_offset = (int(pt_right[0] + offset), int(pt_right[1]))
        cv2.putText(img_matches, str(i), pt_right_offset, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    avg_desparity = avg_desparity/MATCH_POINTS
    return avg_desparity,img_matches,points


def get_depth(disparity,pixel_x,pixel_y,image_width_px,image_height_px):
  
    # Camera parameters (from calibration/spec sheet)
    f_mm = 2.6  # focal length in mm
    sensor_width_mm = 3.6  # sensor width in mm
    sensor_height_mm = 2.7  # sensor height in mm 
    baseline_m = 60/1000

    # Convert focal length to pixels
    fx = (f_mm * image_width_px) / sensor_width_mm
    fy = (f_mm * image_height_px) / sensor_height_mm


    u0 = 2
    v0 = 1
    x_L = pixel_x-u0
    y_L = pixel_y-v0
    # Depth (Z)
    Z = (fx * baseline_m) / disparity
    X = x_L * Z / fx
    Y = y_L * Z / fy

    position_3D = (round(X, 3), round(Y, 3), round(Z, 3))
    print("2D position in image:",(u0,v0))
    print("3D position :",position_3D)



import cv2

# Load stereo pair (left & right)
imgL_r = cv2.imread("img_left.jpg", cv2.IMREAD_GRAYSCALE)
imgR_r = cv2.imread("img_right.jpg", cv2.IMREAD_GRAYSCALE)
scale = 0.5
imgL = cv2.resize(imgL_r, (0, 0), fx=scale, fy=scale)
imgR = cv2.resize(imgR_r, (0, 0), fx=scale, fy=scale)


height_px ,width_px = imgL_r.shape
disparity , img_matches,points = feature_match(imgL,imgR)
print(disparity)

x,y = points[0][0]

get_depth(disparity,x,y,image_width_px=width_px,image_height_px=height_px)
# print(imgL.shape)
# print(imgR.shape)
cv2.imshow("IMG",img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows() 