import numpy as np
import cv2

corner_track_params = dict(maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(200, 200), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Points to track
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)

mask = np.zeros_like(prev_frame)

while True:
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_points, None, **lk_params)

    good_new = next_points[status == 1]
    good_prev = prev_points[status == 1]

    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()

        mask = cv2.line(mask, pt1=(int(x_new), int(y_new)), pt2=(int(x_prev), int(y_prev)), color=(0, 255, 0),
                        thickness=3)

        frame = cv2.circle(frame, (int(x_new), int(y_new)), 8, (0, 0, 255), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('tracking', img)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    prev_gray = frame_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()

# ######################## Dense Version ############################
# can follow all points not just 10 as a limit.


cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()

prev_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(frame1)

# grabbing the saturation channel
hsv_mask[:, :, 1] = 255

while True:
    ret, frame2 = cap.read()
    next_img = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)

    hsv_mask[:,:,0] = ang/2

    hsv_mask[:,:,2] = cv2.normalize(mag, None, 0,255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame',bgr)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    prev_img = next_img

cv2.destroyAllWindows()
cap.release()