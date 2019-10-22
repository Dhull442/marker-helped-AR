import cv2
import numpy as np
import math
import os
import time
from objloader_simple import *

from ar_markers.hamming.detect import detect_markers



def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (100,100,0))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    # print(projection)
    return np.dot(camera_parameters, projection)


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

if __name__ == '__main__':
    # matrix of camera parameters (made up but works quite well for me)
    # camera_parameters = np.array([[435.90240479, 0., 276.97807681], [  0.,   532.43017578, 253.91024449], [0, 0, 1]])
    camera_parameters = np.array([[545.72564697, 0., 455.34108576], [0., 465.43661499, 236.53782366], [ 0., 0., 1. ]])
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, 'marker.jpg'), 0)
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)
    projection = None
    homography = None
    final_homography = None
    capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture('http://192.168.43.1:8080/video')
    if capture.isOpened(): # try to get the first frame
        frame_captured, frame = capture.read()
    else:
        frame_captured = False
    h, w, d = frame.shape
    # h /= 2
    # w /= 2
    # resize = cv2.resize(frame, (int(w), int(h))) 
    dst_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    while frame_captured:
        markers = detect_markers(frame)
        if len(markers) < 1:
            pass
        else:
            marker = markers[0]
            marker.highlite_marker(frame)
            h, w = model.shape
            src_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            tmp_pts = np.float32(marker.contours)
            done = []
            for i in range(len(dst_pts)):
                min_idx = 0
                min_val = np.inf
                for j in range(len(tmp_pts)):
                    if j not in done:
                        dist = ((tmp_pts[j][0][0]-dst_pts[i][0][0])**2 + (tmp_pts[j][0][1]-dst_pts[i][0][1])**2 )**(1/2)
                        if dist < min_val:
                            min_idx = j
                            min_val = dist
                done.append(min_idx)
                dst_pts[i][0] = tmp_pts[min_idx][0]
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if homography is not None:
                try:
                # final_homography = homography
                    if final_homography is None:
                        final_homography = homography
                    else:
                        final_homography = final_homography*0.9 + homography*0.1
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, final_homography)
                    # project cube or model
                    frame = render(frame, obj, projection, model, False)
                except:
                    final_homography = None
                    dst_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    # pass
        cv2.imshow('Video Output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_captured, frame = capture.read()

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


