import numpy as np
import cv2

cap = cv2.VideoCapture('sample2.mp4')
_,d = cap.read()
threshold = 200
summ = np.float32(d)
while(1):
    ret, frame = cap.read()
    cv2.imshow('frame1',frame)
    cv2.accumulateWeighted(frame,summ,0.1)
    res1 = cv2.convertScaleAbs(summ)
    sdf = abs(frame - res1)
    for i in range(len(sdf)):
        for j in range(len(sdf[i])):
            if sdf[i][j][0] >= threshold or sdf[i][j][1] >= threshold or sdf[i][j][2] >= threshold :
                sdf[i][j] = [255,255,255]
            else:
                sdf[i][j] = frame[i][j]
    cv2.imshow('frame2',sdf)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
