import numpy as np
import cv2

def dist(u,sigma,x):
    return  np.absolute((x-u)/sigma)


cap = cv2.VideoCapture('sample2.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
T = 0.5
K = 2
k = 2.5
var_high = 10
wt_low = 0.2
alpha = 0.009
ret, frame = cap.read()
gray_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
shape_fr = np.shape(gray_frame)
u_mean=np.empty((shape_fr[0],shape_fr[1],K))
o_variance=np.empty((shape_fr[0],shape_fr[1],K))
wt=np.empty((shape_fr[0],shape_fr[1],K))
for i in range (K):
    u_mean[:,:,i] = np.copy(gray_frame)
    o_variance[:,:,i] = np.full(shape_fr,var_high)
    wt[:,:,i] = np.full(shape_fr,wt_low)

wt[:,:,0] = np.full(shape_fr,0.6)
wt[:,:,1] = np.full(shape_fr,0.4)
o_variance[:,:,0] = np.full(shape_fr,4)
o_variance[:,:,1] = np.full(shape_fr,var_high)

while(1):
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    cv2.imshow('frame',gray_frame)
    fgmask = fgbg.apply(frame)
    shape_fr = np.shape(gray_frame)
    gray_frame_K=np.repeat(gray_frame[:,:,np.newaxis],K,axis =2)
    dist_mat = (dist(u_mean,o_variance,gray_frame_K))
    # print(dist_mat)
    true_false = dist_mat < k*o_variance

    value = wt/o_variance

    true_false_chk = true_false.sum(axis=2)==0
    true_false_chk2 = true_false.sum(axis=2)==2

    wt_save = np.copy(wt)
    u_mean_save  =  np.copy(u_mean)
    o_variance_save = np.copy(o_variance)

    wt = true_false*((1-alpha) * wt + alpha) + (1-true_false) * (1- alpha) * wt

    rho = alpha * 1 / (np.sqrt(2*np.pi) * o_variance) * np.exp(-0.5*(np.square(gray_frame_K-u_mean/o_variance)))

    u_mean = true_false * ( (1-rho) * u_mean + rho * gray_frame_K ) + (1 - true_false) *u_mean

    o_variance =  true_false * (np.sqrt((1-rho)*np.square(o_variance) + rho* np.square((gray_frame_K-u_mean)))) + (1 - true_false) * o_variance

    wt[:,:,1] = (true_false_chk2)*(1 - alpha) * wt_save[:,:,1] + (1 - true_false_chk2) * wt[:,:,1]
    u_mean[:,:,1] = (true_false_chk2)*u_mean_save[:,:,1] + (1 - true_false_chk2) * u_mean[:,:,1]
    o_variance[:,:,1] = (true_false_chk2)*o_variance_save[:,:,1] + (1 - true_false_chk2) * o_variance[:,:,1]


    # print(true_false)
    # cout = np.bincount(np.reshape(true_false_chk,[shape_fr[0]*shape_fr[1]]))
    # print(cout[0])
    # print(cout[1])

    first_or_second = (wt[:,:,0] - wt[:,:,1])>0



    wt[:,:,0] = (1-first_or_second)*true_false_chk * wt_low + first_or_second*(1 - true_false_chk) * wt[:,:,0]
    u_mean[:,:,0] = (1-first_or_second)*true_false_chk * gray_frame + first_or_second*(1 - true_false_chk) * u_mean[:,:,0]
    o_variance[:,:,0] = (1-first_or_second)*true_false_chk * var_high + first_or_second*(1 - true_false_chk) * o_variance[:,:,0]

    wt[:,:,1] = (1-first_or_second)*true_false_chk * wt_low + first_or_second*(1 - true_false_chk) * wt[:,:,1]
    u_mean[:,:,1] = (1-first_or_second)*true_false_chk * gray_frame + first_or_second*(1 - true_false_chk) * u_mean[:,:,1]
    o_variance[:,:,1] = (1-first_or_second)*true_false_chk * var_high + first_or_second*(1 - true_false_chk) * o_variance[:,:,1]


    f_o_s = (value[:,:,0] - value[:,:,1])>0

    # final_ans = true_false_chk * 255 + (1-true_false_chk)((true_false[:,:,0]*f_o_s)*0 + true_false[:,:,1]*(1 - f_o_s)*0 + 255)
    final_ans = true_false_chk * 1 + (1-true_false_chk)*(true_false[:,:,0]*(1-f_o_s)*1 + (1 - true_false[:,:,0])*true_false[:,:,1] * (f_o_s)*1 )
    ans = np.array(final_ans*255,dtype = np.uint8)
    cv2.imshow('ans',ans)
    cv2.imshow('mog',fgmask)
    cou = np.bincount(np.reshape(final_ans,[shape_fr[0]*shape_fr[1]]))
    # print(cou[0])
    # print(cou[1])
    fg = cv2.waitKey(30) & 0xff
    if fg == 27:
        break

cap.release()
cv2.destroyAllWindows()
