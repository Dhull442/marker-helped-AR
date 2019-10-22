import cv2
import numpy as np
import math
import os
import time
from objloader_simple import *

from ar_markers.hamming.detect import detect_markers



def render(img, obj, projection, other_projection, model, other_model, counter, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = other_model.shape
    centroid = np.asarray([(w-1)/2.0, (h-1)/2.0, 0.]).reshape(-1,1,3)
    centroid2 = cv2.perspectiveTransform(centroid, other_projection)
    h, w = model.shape
    centroid = np.asarray([(w-1)/2.0, (h-1)/2.0, 0.]).reshape(-1,1,3)
    centroid1 = cv2.perspectiveTransform(centroid, projection)

    vec_base = centroid2 - centroid1
    dist = (vec_base**2).sum()**0.5
    vec_base = vec_base / (vec_base**2).sum()**0.5

    # origframe = copy.deepcopy(img)
    # out = cv2.VideoWriter('vid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (frame_width,frame_height))
    # while True:
        # img = copy.deepcopy(origframe)
    vec = vec_base*counter
    all_points_x = []
    all_points_y = []
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        # print(points.shape)
        points = np.dot(points, scale_matrix)
        # translation = np.matmul(np.eye(3),vec)
        # translation = np.asarray([[vec[0],0,0],[0,vec[1],0],[0,0,vec[2]]])
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        # points = points + (vec * (i/1000) * length)
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        dst += vec
        # print(dst)
        # exit(1)
        imgpts = np.int32(dst)
        for point in imgpts:
            all_points_x.append(point[0][0])
            all_points_y.append(point[0][1])
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (100,100,0))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)
    errX = np.mean(all_points_x)-centroid2[0][0][1]
    errY = np.mean(all_points_y)-centroid2[0][0][1]
    # dist = ((errX**2 + errY**2)**0.5)
    # print(dist)
    if(dist < counter):
        return (True,img)
    return (False,img)
    # cv2.imwrite('out/ans'+str(counter)+'.jpg',img)
    # out.write(img)
    # print("frame written")
    # out.release()
    # print("Done")

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
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, 'marker.jpg'), 0)
    kp_model, des_model = orb.detectAndCompute(model, None)
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)
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
    out = cv2.VideoWriter('vid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
    dst_pts_1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    while frame_captured:
        markers = detect_markers(frame)
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
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

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

            # prev_dist = 9999999
            if homography is not None and other_homography is not None:
                # try:
                if final_homography is None:
                    final_homography = homography
                else:
                    final_homography = final_homography*0.9 + homography*0.1
                # obtain 3D projection matrix from homography matrix and camera parameters
                projection = projection_matrix(camera_parameters, final_homography)
                other_projection = projection_matrix(camera_parameters, other_homography)
                counter += 2
                # project cube or model
                # frame = render(frame, obj, projection, model, False)
                brek,frame = render(frame,obj,projection, other_projection, model, model,counter, False)
                out.write(frame)
                if(brek):
                    break
                # except:
                #     final_homography = None
                #     pass
        cv2.imshow('Video Output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_captured, frame = capture.read()

    # When everything done, release the capture
    capture.release()
    out.release()
    cv2.destroyAllWindows()


