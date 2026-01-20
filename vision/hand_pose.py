import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import mediapipe as mp
from stereo_camera.camera.calibration import Calibration
from cvutils import triangulate_with_DLT,triangulate_with_stereopsis
import math
import numpy as np
import os
os.environ["GLOG_minloglevel"] = "2" 

def distance(point1,point2):
    # sqrt((x2-x1)**2 + (y2-y1)**2)
    return math.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)
        
def get_theta(point1,point2,point3):
    a = distance(point1,point2)
    b = distance(point2,point3)
    c = distance(point1,point3)
    
    # Cosine rule 
    theta = math.acos((c**2 - b**2 - a**2)/2*a*b)

    return theta

class HandPose:
    def __init__(self,three_D=False):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.frame_shape = (640, 480)
        self.fig = None
        self.fingers:list[Axes3D] = [0]*21


        # create hand keypoints detector object.
        self.hand_l = self.mp_hands.Hands(min_detection_confidence=0.5,
                                        max_num_hands=1, min_tracking_confidence=0.5)
        self.hand_r = self.mp_hands.Hands(min_detection_confidence=0.5,
                                        max_num_hands=1, min_tracking_confidence=0.5)

        # projection matrices
        self.P1 = Calibration.get_projection_matrix()
        self.P2 = Calibration.get_projection_matrix(1)

        self.hand =  self.mp_hands.Hands(min_detection_confidence=0.5,max_num_hands=1, min_tracking_confidence=0.5)

    def extract_keypoints(self, frame_l, frame_r):
        # crop to 720x720.
        # Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        # if frame_l.shape[1] != 720:
        #     frame_l = frame_l[:, self.frame_shape[1]//2 - self.frame_shape[0] //2:self.frame_shape[1]//2 + self.frame_shape[0]//2]
        #     frame_r = frame_r[:, self.frame_shape[1]//2 - self.frame_shape[0] //2:self.frame_shape[1]//2 + self.frame_shape[0]//2]

        # the BGR image to RGB.
        frame_l = cv.cvtColor(frame_l, cv.COLOR_BGR2RGB)
        frame_r = cv.cvtColor(frame_r, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        frame_l.flags.writeable = False
        frame_r.flags.writeable = False

        results0 = self.hand_l.process(frame_l)
        results1 = self.hand_r.process(frame_r)

        # prepare list of hand keypoints of this frame
        # frame0 kpts
        frame_l_keypoints = []
        if results0.multi_hand_landmarks:
            for hand_landmarks in results0.multi_hand_landmarks:
                for p in range(21):
                    # print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    pxl_x = int(
                        round(frame_l.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(
                        round(frame_l.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame_l_keypoints.append(kpts)

        # no keypoints found in frame:
        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame_l_keypoints = [[-1, -1]]*21

        # frame_r kpts
        frame_r_keypoints = []
        if results1.multi_hand_landmarks:
            for hand_landmarks in results1.multi_hand_landmarks:
                for p in range(21):
                    # print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    pxl_x = int(
                        round(frame_r.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(
                        round(frame_r.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame_r_keypoints.append(kpts)

        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame_r_keypoints = [[-1, -1]]*21
        
        return  (frame_l_keypoints,frame_r_keypoints) 

    def get_3D_cords(self,keypoints_l,keypoints_r):
         # calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(keypoints_l, keypoints_r):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                # calculate 3d position of keypoint
                _p3d = triangulate_with_DLT(self.P1, self.P2, uv1, uv2)
            frame_p3ds.append(_p3d)

        frame_p3ds = np.array(frame_p3ds).reshape((21, 3))
        return frame_p3ds

    def visualize_3d(self,p3ds):
        """Apply coordinate rotations to point z axis as up"""
        Rz = np.array(([[0., -1., 0.],
                        [1.,  0., 0.],
                        [0.,  0., 1.]]))
        Rx = np.array(([[1.,  0.,  0.],
                        [0., -1.,  0.],
                        [0.,  0., -1.]]))

        p3ds_rotated = []
        for kpt in p3ds:
            kpt_rotated = Rz @ Rx @ kpt
            p3ds_rotated.append(kpt_rotated)

        """this contains 3d points of each frame"""
        kpts3d_b = np.array(p3ds_rotated)
    
        # Shift the origin to wrist point
        kpts3d = kpts3d_b - kpts3d_b[[0],:]

        """Now visualize in 3D"""
        thumb_f = [[0, 1], [1, 2], [2, 3], [3, 4]]
        index_f = [[0, 5], [5, 6], [6, 7], [7, 8]]
        middle_f = [[0, 9], [9, 10], [10, 11], [11, 12]]
        ring_f = [[0, 13], [13, 14], [14, 15], [15, 16]]
        pinkie_f = [[0, 17], [17, 18], [18, 19], [19, 20]]
        fingers = [pinkie_f, ring_f, middle_f, index_f, thumb_f]
        fingers_colors = ['red', 'blue', 'green', 'black', 'orange']

        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

            for finger, finger_color in zip(fingers, fingers_colors):
                for _c in finger:
                    # print(_c,kpts3d)
                    self.fingers[_c[1]] = self.ax.plot(xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]],
                            ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]], 
                            zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]],
                            linewidth=4,
                            c=finger_color)[0]
                plt.pause(0.01)
        else:
            for finger, finger_color in zip(fingers, fingers_colors):
                for _c in finger:
                    # print(_c,kpts3d)
                    self.fingers[_c[1]].set_data_3d([kpts3d[_c[0], 0], kpts3d[_c[1], 0]],
                                                            [kpts3d[_c[0], 1], kpts3d[_c[1], 1]], 
                                                            [kpts3d[_c[0], 2], kpts3d[_c[1], 2]])
                plt.draw()
                plt.pause(0.01)
            # # draw axes
            # self.ax.plot(xs=[0, 5], ys=[0, 0], zs=[0, 0], linewidth=2, color='red')
            # self.ax.plot(xs=[0, 0], ys=[0, 5], zs=[0, 0], linewidth=2, color='blue')
            # self.ax.plot(xs=[0, 0], ys=[0, 0], zs=[0, 5],
            #         linewidth=2, color='black')

            self.ax.set_xlim3d(-25, 25)
            self.ax.set_xlabel('x')
            self.ax.set_ylim3d(-25, 25)
            self.ax.set_ylabel('y')
            self.ax.set_zlim3d(-25, 25)
            self.ax.set_zlabel('z')
            # plt.savefig('figs/fig_' + str(i) + '.png')
            plt.pause(0.01)
        

    def get_fingure_angles(self,keypoints):
        finger1 = [0,1,2,3]
        finger2 = [0,5,6,7]
        finger3 = [0,9,10,11]

        angles = []

        for i in range(2):
            # For finger 1
            point11 = keypoints[finger1[i]]
            point12 = keypoints[finger1[i+1]]
            point13 = keypoints[finger1[i+2]]

            theta = get_theta(point11,point12,point13)
            angles.append(theta)


        for i in range(2):
            # For finger 2
            point21 = keypoints[finger2[i]]
            point22 = keypoints[finger2[i+1]]
            point23 = keypoints[finger2[i+2]]

            theta = get_theta(point21,point22,point23)
            angles.append(theta)


        for i in range(2):
            # For finger 3
            point31 = keypoints[finger3[i]]
            point32 = keypoints[finger3[i+1]]
            point33 = keypoints[finger3[i+2]]

            theta = get_theta(point31,point32,point33)
            angles.append(theta)

        return angles
    def get_fingure_angles_3D(self,keypoints):
        pass

    def get_pose(self,frame):
        results = self.hand_l.process(frame)

        keypoints = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for p in range(21):
                    pxl_x = int(
                        round(frame.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(
                        round(frame.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = (pxl_x, pxl_y)
                    keypoints.append(kpts)

        theta = self.get_fingure_angles(keypoints)
        return theta,keypoints[0]

    def get_pose_3D(self,frame_l, frame_r,visualize=False):
        keypoints_2D = self.extract_keypoints(frame_l, frame_r)
        keypoints_3D = self.get_3D_cords(*keypoints_2D)
        # theta = self.get_fingure_angles_3D(keypoints_3D)

        if visualize:
            self.visualize_3d(keypoints_3D)
            
        # return theta,keypoints_3D[0]


    def draw(self,img_left,img_right):
        # Process frame
        frame_l = cv.cvtColor(img_left, cv.COLOR_BGR2RGB)
        frame_r = cv.cvtColor(img_right, cv.COLOR_BGR2RGB)
         # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        # frame_l.flags.writeable = False
        # frame_r.flags.writeable = False
        results_l = self.hand_l.process(frame_l)
        results_r = self.hand_r.process(frame_r)

        # Draw hand landmarks
        if results_l.multi_hand_landmarks:
            for hand_landmarks in results_l.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_l,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )
        # Draw hand landmarks
        if results_r.multi_hand_landmarks:
            print("JIii")
            for hand_landmarks in results_r.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_r,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )
        frame_l = cv.cvtColor(frame_l, cv.COLOR_RGB2BGR)
        frame_r = cv.cvtColor(frame_r, cv.COLOR_RGB2BGR)
        return frame_l,frame_r
   
if __name__ == "__main__":
    # hp = HandPose()
    # img_ls = cv.imread(r"data\img_left.jpg")
    # img_rs = cv.imread(r"data\img_right.jpg")
    # img_ls = cv.resize(img_ls, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    # img_rs = cv.resize(img_rs, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    # keypts_l,keypts_r = hp.extract_keypoints(img_ls,img_rs)
    # points3D = hp.get_3D_cords(keypts_l,keypts_r)
    # hp.visualize_3d(points3D)
    # print(f"Keypoint Left Image: {keypts_l[0]}")
    # print(f"Keypoint Right Image: {keypts_r[0]}")
    # print(f"Keypoint in 3D (DLT): {point3D_DLT}")
    # print(f"Keypoint in 3D (Stereopsis): {point3D_stereo}")
    # cv.circle(img_ls, keypts_l[0], radius=5, color=(0, 0, 255), thickness=-1)  # Red filled circles
    # # cv.circle(img_rs, keypts_r[0], radius=5, color=(0, 0, 255), thickness=-1)  # Red filled circles
    # imgl,imgr = hp.draw(img_ls,img_rs)
    # stereo_image = np.hstack((imgl,imgr))  # Or use cv2.hconcat([left_image, right_image])
    # # Show the stereo image
    # show_cv = False

    # if show_cv:
    #     cv.imshow("Stereo Image (Left | Right)", stereo_image)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
    # else:
    #     fig = plt.figure(figsize=(18,9))
    #     ax1 = fig.add_subplot(121)
    #     ax2 = fig.add_subplot(122)
    #     ax1.imshow(cv.cvtColor(imgl,cv.COLOR_RGB2BGR))
    #     ax1.grid(visible=True)
    #     ax2.imshow(cv.cvtColor(imgr,cv.COLOR_RGB2BGR))
    #     ax2.grid(visible=True)
    #     plt.show()
    import torch
    import torchvision
    from torchvision.transforms import functional as F
    from PIL import Image
    import matplotlib.pyplot as plt



    # Load pre-trained Mask R-CNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    cap_l = cv.VideoCapture(r"data\v_left.mp4")
    # cap_r = cv.VideoCapture(r"data\v_right.mp4")
    plt.ion()
    while cap_l.isOpened():
        flag1,frame_l = cap_l.read()
        # flag2,frame_r = cap_r.read()
        flag2 = True

        if (not flag1) or (not flag2):
            break
        img_ls = cv.resize(frame_l, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        # img_rs = cv.resize(frame_r, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        # Load image
        # img = Image.open(r'data\img_left.jpg').convert("RGB")
        img_tensor = F.to_tensor(img_ls)
        with torch.no_grad():
            prediction = model([img_tensor])[0]

        # Visualize masks
        plt.imshow(img_ls)
        for i in range(len(prediction['boxes'])):
            mask = prediction['masks'][i, 0].mul(255).byte().cpu().numpy()
            plt.imshow(mask, alpha=0.5)
        plt.axis('off')
        plt.pause(0.05)

        # keypts_l,keypts_r = hp.extract_keypoints(img_ls,img_rs)
        # points3D = hp.get_3D_cords(keypts_l,keypts_r)
        # print(".",end=" ")
        # hp.visualize_3d(points3D)
        # plt.pause(0.01)

        # imgl,imgr = hp.draw(img_ls,img_rs)
        # stereo_image = np.hstack((imgl,imgr))  # Or use cv2.hconcat([left_image, right_image])
        # #Show the stereo image
        # cv.imshow("Stereo Image (Left | Right)", stereo_image)

        # if cv.waitKey(500) & 0xFF == ord('q'):
        #     break
    plt.ioff()
    plt.show()
    cap_l.release()
    # cap_r.release()
    # cv.destroyAllWindows()
    