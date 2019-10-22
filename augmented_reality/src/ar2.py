
# Useful links
# http://www.pygame.org/wiki/OBJFileLoader
# https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
# https://clara.io/library

# TODO -> Implement command line arguments (scale, model and object to be projected)
#      -> Refactor and organize code (proper funcition definition and separation, classes, error handling...)

import argparse

import cv2
import numpy as np
import math
import os
import copy
from objloader_simple import *

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 10


def main():
    """
    This functions loads the target surface image,
    """
    homography = None
    # matrix of camera parameters (made up but works quite well for me)
    camera_parameters = np.array([[435.90240479, 0., 276.97807681], [0., 532.43017578, 253.91024449], [0, 0, 1]])
    # create ORB keypoint detector
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, 'reference/marker_websak_1.jpg'), 0)
    other_model = cv2.imread(os.path.join(dir_name, 'reference/marker_websak_0.jpg'), 0)
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    kp_other_model, des_other_model = orb.detectAndCompute(other_model, None)
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)
    # init video capture
    cap = cv2.VideoCapture(0)
    # frame = cv2.imread('pic0.jpg')
    counter = 0
    frame_height = int(cap.get(4))
    frame_width = int(cap.get(3))
    out = cv2.VideoWriter('vid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
    # h = np.eye(3)
    while True:
    #     # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return
        # find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # match frame descriptors with model descriptors
        matches = bf.match(des_model, des_frame)
        other_matches = bf.match(des_other_model, des_frame)
        # sort them in the order of their distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)
        other_matches = sorted(other_matches, key=lambda x: x.distance)
        dist = []
        for m in matches:
            dist.append(m.distance)
        dist = np.asarray(dist)
        good = np.median(dist)<40
        dist = []
        for m in matches:
            dist.append(m.distance)
        dist = np.asarray(dist)
        good = good and (np.median(dist)<40)
        # print(np.median(dist))
        # differenciate between source points and destination points
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # compute Homography
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        src_pts = np.float32([kp_other_model[m.queryIdx].pt for m in other_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in other_matches]).reshape(-1, 1, 2)
        # compute Homography
        other_homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Draw a rectangle and get centroid that marks the found models in the frame
        h, w = model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, homography)
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        h, w = other_model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, other_homography)
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        prev_dist = 9999999
        if homography is not None and good:
            # try:
            # print(homography)
            # obtain 3D projection matrix from homography matrix and camera parameters
            projection = projection_matrix(camera_parameters, homography)
            other_projection = projection_matrix(camera_parameters, other_homography)
            # project cube or model
            counter += 2
            # if(counter)
            brek,frame = render(frame,obj,projection, other_projection, model, other_model,counter, False)
            out.write(frame)
            if(brek):
                break
            # except:
            #     pass
        # frame = render(frame, obj,projection, other_projection, model, other_model, 100, False)
        # cv2.imwrite('ans.jpg',frame)
        # draw first 10 matches.
        if args.matches and good:
            frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
        # show result
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return 0


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
    print(dist)
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


# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
# TODO jgallostraa -> add support for model specification
#parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
