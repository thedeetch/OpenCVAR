import numpy as np
import scipy as sp
import scipy.signal
import cv2

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
OUTPUT_FILE='output.mp4'
MATCH_IMAGE='match.jpg'

DRAW = 'matches' #'keypoints'

capture = cv2.VideoCapture(0)
capture.set(1, VIDEO_FPS)
capture.set(3, VIDEO_WIDTH)
capture.set(4, VIDEO_HEIGHT)

# overlay = cv2.VideoCapture('ccau_512kb.mp4')
# overlay.set(1, VIDEO_FPS)

fourcc = cv2.cv.CV_FOURCC(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, VIDEO_FPS / 2.0, (VIDEO_WIDTH, VIDEO_HEIGHT))

match_img = cv2.imread('githuboctocat.jpeg')
MATCH_KP, MATCH_DESC = findKeyPoints(match_img)

while(True):
    ret, frame = capture.read()
    # retOverlay, frameOverlay = overlay.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    frame_kp, frame_desc = findKeyPoints(frame)

    if DRAW == 'keypoints':
        frame = cv2.drawKeypoints(frame, frame_kp)
    elif DRAW == 'matches':
        frame_kp, frame_desc = findKeyPoints(frame)
        matches = findMatches(frame_desc, MATCH_DESC)
        # frame = cv2.drawKeypoints(frame, matches)
        points = [frame_kp[m.queryIdx] for m in matches]
        frame = cv2.drawKeypoints(frame, points)
        # frame = cv2.drawMatches(frame,frame_kp,match_img,match_kp,matches[:10], flags=2)

    cv2.imshow('frame',frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
overlay.release()
out.release()
cv2.destroyAllWindows()
