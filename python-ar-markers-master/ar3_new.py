import cv2
import numpy as np
import math
import os
import copy
import time

from ar_markers.hamming.detect import detect_markers


def sqrt(x):
    return x**(0.5)
class Ball:
    def __init__(self):
        self.r = 10
        self.x = 10
        self.y = 10
        self.speed = 2
        self.angle = [1/sqrt(2),1/sqrt(2)]
        self.refjust = False
        self.times = 0
    def dist(self,line):
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[1][0]
        y2 = line[1][1]
        # print(x1,y1,x2,y2)
        val = abs(((y2-y1)*self.x) - ((x2-x1)*self.y) + (x2*y1-x1*y2) )/sqrt((y2-y1)**2 + (x2-x1)**2)
        # print(val)
        return val
    def check(self,line):
        #check if the point is on segment
        x = (line[0][0]+line[1][0])/2
        y = (line[0][1]+line[1][1])/2
        d = sqrt((line[0][1]-line[1][1])**2 + (line[0][0]-line[1][0])**2) / 2
        dp = sqrt((self.y - y)**2 + (self.x - x)**2)
        return dp <= d

    def changepos(self,line0,line1):
        dst = []
        # print(self.refjust,self.times)
        if(self.times > 50):
            self.refjust = False
            self.times = 0
        if(self.refjust):
            self.times += 1
        else:
            if(self.dist(line0)<self.r and self.check(line0)):
                # print("ref")
                self.refjust = True
                dst = line0
            elif(self.dist(line1)<self.r and self.check(line1)):
                self.refjust = True
                dst = line1
            if(len(dst)>0 and self.refjust):
                self.speed+=0.5
                sl1 = (dst[0][1] - dst[1][1])/(dst[0][0] - dst[1][0])
                sl2 = self.angle[1]/self.angle[0]
                # m1 = [(dst[0][0] - dst[1][0]),(dst[0][1] - dst[1][1])]
                sl3 = (2*sl1 + sl2*(sl1**2) - sl2)/(2*sl1*sl2-sl1**2 + 1)
                angle = [math.cos(math.atan(sl3)),math.sin(math.atan(sl3))]
                mc = [sl1,0.]
                mc[1] = dst[0][1] - mc[0]*dst[0][0]
                nx = self.x + 50*angle[0]
                ny = self.y + 50*angle[1]
                if((self.x*mc[0]-self.y+mc[1])*(nx*mc[0]-ny+mc[1]) > 0):
                    self.angle[0] = angle[0]
                    self.angle[1] = angle[1]
                else:
                    self.angle[0] = -1*angle[0]
                    self.angle[1] = -1*angle[1]
                # print(self.angle)
        self.update();
    def update(self):
        self.x = self.x + self.angle[0] * self.speed
        self.y = self.y + self.angle[1] * self.speed


if __name__ == '__main__':
    # matrix of camera parameters (made up but works quite well for me)
    # camera_parameters = np.array([[435.90240479, 0., 276.97807681], [  0.,   532.43017578, 253.91024449], [0, 0, 1]])
    camera_parameters = np.array([[545.72564697, 0., 455.34108576], [0., 465.43661499, 236.53782366], [ 0., 0., 1. ]])
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ball = Ball();
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, 'marker.jpg'), 0)
    kp_model, des_model = orb.detectAndCompute(model, None)
    # Load 3D model from OBJ file
    homography = None
    final_homography = None
    capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture('http://192.168.43.1:8080/video')
    if capture.isOpened(): # try to get the first frame
        frame_captured, frame = capture.read()
    else:
        frame_captured = False
    h, w, d = frame.shape
    counter = 0
    frame_height = int(capture.get(4))
    frame_width = int(capture.get(3))
    out = cv2.VideoWriter('ping_pong.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
    dst_pts_1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    while frame_captured:
        markers = detect_markers(frame)
        line_0 = [[0,5000],[5000,0]]
        line_1 = [[0,3000],[3000,0]]
        if len(markers) < 1:
            pass
        else:
            kp_frame, des_frame = orb.detectAndCompute(frame, None)
            # match frame descriptors with model descriptors
            matches = bf.match(des_model, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)
            dist = []
            for m in matches:
                dist.append(m.distance)
            dist = np.asarray(dist)
            good = np.median(dist)<40
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # compute Homography
            other_homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = model.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, other_homography)
            # frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            marker = markers[0]
            marker.highlite_marker(frame)
            h, w = model.shape
            src_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            tmp_pts = np.float32(marker.contours)
            done = []
            for i in range(len(dst_pts_1)):
                min_idx = 0
                min_val = np.inf
                for j in range(len(tmp_pts)):
                    if j not in done:
                        dist = ((tmp_pts[j][0][0]-dst_pts_1[i][0][0])**2 + (tmp_pts[j][0][1]-dst_pts_1[i][0][1])**2 )**(1/2)
                        if dist < min_val:
                            min_idx = j
                            min_val = dist
                done.append(min_idx)
                dst_pts_1[i][0] = tmp_pts[min_idx][0]
            homography, mask = cv2.findHomography(src_pts, dst_pts_1, cv2.RANSAC, 5.0)

            if homography is not None and other_homography is not None:
                # try:
                # if final_homography is None:
                #     final_homography = homography
                # else:
                #     final_homography = final_homography*0.9 + homography*0.1
                # obtain 3D projection matrix from homography matrix and camera parameters
                h, w = model.shape
                pts = np.float32([[(w-1)/2, 0], [(w-1)/2, h]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, homography)                         ## first marker 2 points to define a ling
                line_0[0][0] = dst[0][0][0]
                line_0[0][1] = dst[0][0][1]
                line_0[1][0] = dst[1][0][0]
                line_0[1][1] = dst[1][0][1]
                frame = cv2.polylines(frame, [np.int32(dst)], True, 200, 3, cv2.LINE_AA)
                dst = cv2.perspectiveTransform(pts, other_homography)                   ## second marker 2 points to define a ling
                line_1[0][0] = dst[0][0][0]
                line_1[0][1] = dst[0][0][1]
                line_1[1][0] = dst[1][0][0]
                line_1[1][1] = dst[1][0][1]
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                # except:
                #     final_homography = None
                #     pass
        ball.changepos(line_0,line_1)
        x = int(ball.x)
        y = int(ball.y)
        if( y < frame.shape[0] and x < frame.shape[1] and x >= 0 and y >= 0 ):
            frame = cv2.circle(frame, (x,y), 10, (25,179,255), -1)
        else:
            break
        cv2.imshow('Video Output',cv2.flip(frame,1))
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_captured, frame = capture.read()

    # When everything done, release the capture
    capture.release()
    out.release()
    cv2.destroyAllWindows()


