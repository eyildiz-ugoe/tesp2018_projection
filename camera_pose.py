import numpy as np
import cv2

MIN_MATCH_COUNT = 10
#just using this variable for testing purposes
projFrame = 'solar_system.jpg'


def get_feature_match(camFrame, projFrame):
    #Get the images and convert them to gray scale
    #img1 = cv2.imread(camFrame)
    img1 = camFrame
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread(projFrame)
    #img2 = projFrame
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Consider trying to actually get sift to work here
    #initialise orb which is similar to sift but free
    orb = cv2.ORB_create()
    #find the keypoints and descriptors using orb
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    #Brute-Force matcher
    #BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)



    return matches, kp1, kp2

#####FLANN based matcher didn't work :( Maybe because I'm using orb instead of sift
    #FLANN based matcher
    #FLANN parameters
    #FLANN_INDEX_LSH = 6
    #index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    #search_params = dict(checks=50) # or pass empty dictionary

    #flann = cv2.FlannBasedMatcher(index_params, search_params)

    #matches = flann.knnMatch(des1, des2, k=2)

    #ratio test which can be found in D.Lowe's paper to store only good matches
    #good = []
    #for m,n in matches:
     #   if m.distance < 0.7*n.distance:
      #      good.append(m)


    #return good, kp1, kp2


def get_homography(matches, kp1, kp2):
    #Apply ratio test to get good matches
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    #check that there is enough matches
    if len(matches)>MIN_MATCH_COUNT:
        #Not really sure what this is doing, what are queryIdx and trainIdx
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        #get the homography matrix
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return matrix
    else:
        #if a homography matrix could not be found return 0
        return 0

#returns the location of the centre of the camera image in the projector image
def virtual_point(camFrame, hgmatrix):
    #get the width and height of the camera image
    #img1 = cv2.imread(camFrame) <<< do I need this?
    h,w,d = camFrame.shape

    #The centre point in the camera image
    pts = np.float32([[w/2,h/2]]).reshape(-1,1,2)
    # find the location of this same point in the projector image
    dst = cv2.perspectiveTransform(pts, hgmatrix)

    return dst

#initialise the webcam
cap = cv2.VideoCapture(1)

while (True): #if this is an infinite loop do all other functions have to be called from here.
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Get the best matching features from the images and the key points from each image
    matches, kp1, kp2 = get_feature_match(frame, projFrame)
    #get the homography matrix from these features
    hgmatrix = get_homography(matches, kp1, kp2)

    display = cv2.imread(projFrame)

    if hgmatrix!=0:
        #get the location of the centre of the camera image in the projector image
        dspPoint = virtual_point(frame, hgmatrix)
        # draw a circle on where we clicked
        cv2.circle(display, dspPoint[0], 3, (0, 0, 255), thickness=-1, lineType=8)  # color BGR




    # Display the resulting projector image with a dot for the camera location
    cv2.imshow('projector', display)
    if cv2.waitKey(1) & 0xFF == 0x1b:  # ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()








#def get_feature_match(camFrame, projFrame):
    #Get the images and convert them to gray scale
    #img1 = cv2.imread(camFrame)
    #gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    #img2 = cv2.imread(projFrame)
    #gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #initialise sift
    #sift = cv2.SIFT()
    #find the keypoints and descriptors using sift
    #kp1, des1 = sift.detectAndCompute(gray1, None)
    #kp2, des2 = sift.detectAndCompute(gray2, None)


    #BFMatcher with default params
    #bf = cv2.BFMatcher()
    #gives k best matches
    #matches = bf.knnMatch(des1, des2, k=2)

    #Apply ratio test from paper by D.Lowe
    #good = []
    #for m, n in matches:
    #    if m.distance < 0.75*n.distance:
    #        good.append([m])

    #return good