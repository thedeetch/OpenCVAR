import numpy as np
import scipy as sp
import scipy.signal
import cv2

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
OUTPUT_FILE='output.mp4'
MATCH_IMAGE='match.jpg'

DRAW = 'image' #'matches' #'keypoints'

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
        x = [int(frame_kp[m.queryIdx].pt[0]) for m in matches]
        y = [int(frame_kp[m.queryIdx].pt[1]) for m in matches]

        if len(points) > 20:
            pt1 = (min(x), min(y))
            pt2 = (max(x), max(y))
            frame = cv2.drawKeypoints(frame, points)
            if max(x) < VIDEO_HEIGHT and max(y) < VIDEO_WIDTH:
                cv2.rectangle(frame, tuple(pt1), tuple(pt2), (255,0,0), 2)
    elif DRAW == 'image':
        frame_kp, frame_desc = findKeyPoints(frame)
        matches = findMatches(frame_desc, MATCH_DESC)
        # frame = cv2.drawKeypoints(frame, matches)
        points = [frame_kp[m.queryIdx] for m in matches]
        x = [int(frame_kp[m.queryIdx].pt[0]) for m in matches]
        y = [int(frame_kp[m.queryIdx].pt[1]) for m in matches]

        if len(points) > 20:
            pt1 = (min(x), min(y))
            pt2 = (max(x), max(y))
            frame = cv2.drawKeypoints(frame, points)
            if max(x) < VIDEO_HEIGHT and max(y) < VIDEO_WIDTH:
                cv2.rectangle(frame, tuple(pt1), tuple(pt2), (255,0,0), 2)
            image_1_points = np.float32([ frame_kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            image_2_points = np.float32([ MATCH_KP[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
            image_1_corners = getImageCorners(frame)
            image_2_corners = getImageCorners(match_img)

            M, mask = cv2.findHomography(image_2_points, image_1_points, cv2.RANSAC, 5.0)
            t = cv2.perspectiveTransform(image_2_corners, M)

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
            warped_image = cv2.warpPerspective(match_img,dotProduct,(x_max - x_min, y_max - y_min))

            # frame = warped_image

            frame[point[1]:point[1] + frame.shape[0],
                point[0]:point[0] + frame.shape[1]] = warped_image

    cv2.imshow('frame',frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
# overlay.release()
out.release()
cv2.destroyAllWindows()
