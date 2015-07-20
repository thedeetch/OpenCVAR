import numpy as np
import scipy as sp
import scipy.signal
import cv2
import argparse

parser = argparse.ArgumentParser(description='Augmented reality using OpenCV.')
parser.add_argument('-i','--input', help='Image to find (default=match.png)', default='match.png')
parser.add_argument('-p','--paste', help='Image to paste over found feature')
parser.add_argument('-v','--pastevideo', help='Video to paste over found feature')
parser.add_argument('-o','--output',help='Output video file name (default=output.mp4)', default='output.mp4')
parser.add_argument('-b','--blur',help='Blur found feature (default=False)', action='store_true')
parser.add_argument('-r','--rectangle',help='Draw rectange around feature (default=False)', action='store_true')
parser.add_argument('-k','--keypoints',help='Draw all feature keypoints or just matches', choices=['keypoints', 'matches'])
parser.add_argument('-n','--nummatches',help='Minimum number of matches (default=20)', default=20)

args = parser.parse_args()
# print args
def getImageCorners(image):
  corners = np.zeros((4, 1, 2), dtype=np.float32)
  # WRITE YOUR CODE HERE

  corners[0] = [0,0]
  corners[1] = [0,image.shape[0]]
  corners[2] = [image.shape[1],0]
  corners[3] = [image.shape[1],image.shape[0]]

  return corners

def findKeyPoints(image):
    # orb = cv2.ORB_create
    orb = cv2.ORB(1000)
    return orb.detectAndCompute(image, None)

def findMatches(desc1, desc2):
    index_params= dict(algorithm = 6,
                     table_number = 12,#6, # 12
                     key_size = 20,#12,     # 20
                     multi_probe_level = 2)#1) #2

    search_params = dict(checks=500)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1,desc2, 2)

    good = []
    # ratio test as per Lowe's paper
    for m in matches:
        if len(m) > 0 and m[0].distance < 0.7*m[-1].distance:
            good.append(m[0])
            # matchesMask[i]=[1,0]
    return good

VIDEO_FPS=20.0
VIDEO_WIDTH=640
VIDEO_HEIGHT=480
OVERLAY_WIDTH=320
OVERLAY_HEIGHT=240

capture = cv2.VideoCapture(0)
capture.set(1, VIDEO_FPS)
capture.set(3, VIDEO_WIDTH)
capture.set(4, VIDEO_HEIGHT)

fourcc = cv2.cv.CV_FOURCC(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, VIDEO_FPS / 2.0, (VIDEO_WIDTH, VIDEO_HEIGHT))

match_img = cv2.imread(args.input)
MATCH_KP, MATCH_DESC = findKeyPoints(match_img)

paste_img = np.zeros((0,0,0))
if args.paste:
    paste_img = cv2.imread(args.paste)
elif args.pastevideo:
    overlay = cv2.VideoCapture(args.pastevideo)
    overlay.set(1, VIDEO_FPS)

while(True):
    ret, frame = capture.read()
    # retOverlay, frameOverlay = overlay.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_kp, frame_desc = findKeyPoints(frame)
    matches = findMatches(frame_desc, MATCH_DESC)
    # frame = cv2.drawKeypoints(frame, matches)
    points = [frame_kp[m.queryIdx] for m in matches]
    x = [int(frame_kp[m.queryIdx].pt[0]) for m in matches]
    y = [int(frame_kp[m.queryIdx].pt[1]) for m in matches]

    pt1=(0,0)
    pt2=(0,0)

    image_1_corners = getImageCorners(frame)
    image_2_corners = getImageCorners(match_img)

    # print tuple(image_1_corners[2][0])
    FONT_SIZE=0.5
    size = cv2.getTextSize("Hello World!!!", cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, 1)
    cv2.putText(frame,"keypoints: " + str(len(MATCH_KP)) + " matches: " + str(len(matches)), tuple(image_1_corners[1][0]), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, [255,255,255])
    # cv2.putText(frame,, tuple(image_1_corners[1][0]), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, [255,255,255])

    if args.keypoints == 'keypoints':
        frame = cv2.drawKeypoints(frame, frame_kp)

    if len(points) >= args.nummatches:
        pt1 = (min(x), min(y))
        pt2 = (max(x), max(y))

        if args.keypoints == 'matches':
            frame = cv2.drawKeypoints(frame, points)
        if args.rectangle:
            cv2.rectangle(frame, tuple(pt1), tuple(pt2), (0,255,0), 2)
        if args.paste or args.pastevideo:
            if args.pastevideo:
                ret, paste_img = overlay.read()

            image_1_points = np.float32([ frame_kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            image_2_points = np.float32([ MATCH_KP[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
            paste_corners = getImageCorners(paste_img)
            M, mask = cv2.findHomography(image_2_points, image_1_points, cv2.RANSAC, 5.0)
            t = cv2.perspectiveTransform(paste_corners, M)

            corners = np.concatenate((t, image_1_corners))

            corners = np.min(corners, axis=1)
            xCol = corners[:,:1]
            yCol = corners[:,1:]
            x_min = np.min(xCol)
            x_max = np.max(xCol)
            y_min = np.min(yCol)
            y_max = np.max(yCol)

            translation = np.array([[1, 0, -1 * x_min],
                                  [0, 1, -1 * y_min],
                                  [0, 0, 1]])

            point = (-1 * x_min, -1 * y_min)
            dotProduct = np.dot(translation, M)
            warped_image = cv2.warpPerspective(paste_img,dotProduct,(x_max - x_min, y_max - y_min), frame, borderMode=cv2.BORDER_TRANSPARENT)
        if args.blur:
            match = frame[pt1[1]:pt2[1],pt1[0]:pt2[0]]
            match = cv2.GaussianBlur(match,(23, 23), 30)
            frame[pt1[1]:pt2[1],pt1[0]:pt2[0]] = match

    cv2.imshow('frame',frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
overlay.release()
out.release()
cv2.destroyAllWindows()
