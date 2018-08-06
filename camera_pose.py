import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
#just using this variable for testing purposes
projFrame = 'solar_system.jpg'

#Function which takes the camera image and the projector frame and finds the keypoints in these images
#It then finds the matches
#Returns matches, and keypoints for both images
def get_feature_match(camFrame, projFrame):
    #Get the images and convert them to gray scale
    #img1 = cv2.imread(camFrame)
    img1 = camFrame
    camera_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    #switch these round later, this was just for testing purposes
    img2 = cv2.imread(projFrame)
    #img2 = projFrame
    proj_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Consider trying to actually get sift to work here
    #initialise orb which is similar to sift but free
    orb = cv2.ORB_create()
    #find the keypoints and descriptors using orb
    camera_kp, des1 = orb.detectAndCompute(camera_gray, None)
    proj_kp, des2 = orb.detectAndCompute(proj_gray, None)

    #Brute-Force matcher
    #BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    #Test for the matcher
    """good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,  flags=2)
    plt.imshow(img3), plt.show()"""

    return matches, camera_kp, proj_kp

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

#Function to find the homography matrix which transforms from the camera image to the projector image
def get_homography(matches, camera_kp, proj_kp):
    homography_matrix = np.array[]
    #Apply ratio test to get good matches
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    #check that there is enough matches
    if len(matches)>MIN_MATCH_COUNT:
        #Not really sure what this is doing, what are queryIdx and trainIdx
        src_pts = np.float32([proj_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([camera_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        #get the homography matrix
        homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return homography_matrix
    else:
        print()
        #if a homography matrix could not be found return 0. I am finding it hard to test for this.
        #Need something we can test for later in cases where there weren;t enough matches
        #INitialise the matrix with zeros at the start of this function and just return the empty matrix if it goes into
        #the else case
        #Should also set mask = None or something so there are no errors  in the future
        return np.array(np.array[0, 0])

#returns the location of the centre of the camera image in the projector image
def virtual_point(camFrame, hgmatrix):
    #get the width and height of the camera image
    #img1 = cv2.imread(camFrame) <<< do I need this?
    h,w,d = camFrame.shape

    #The centre point in the camera image
    #pts = np.float32([[w/2,h/2]]).reshape(-1,1,2)
    pts = np.array([[w/2, h/2]], dtype='float32')
    pts = np.array([pts])
    # find the location of this same point in the projector image
    dst = cv2.perspectiveTransform(pts, hgmatrix)

    return dst



###################Main piece of code


#initialise the webcam
cap = cv2.VideoCapture(1)
ret = False
while ret == False:
    ret, frame = cap.read()
    matches, kp1, kp2 = get_feature_match(frame, projFrame)

    hgmatrix = get_homography(matches, kp1, kp2)
    print(hgmatrix)



"""while (True): #if this is an infinite loop do all other functions have to be called from here.
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Get the best matching features from the images and the key points from each image
    matches, kp1, kp2 = get_feature_match(frame, projFrame)
    #get the homography matrix from these features
    hgmatrix = get_homography(matches, kp1, kp2)

    display = cv2.imread(projFrame)

    #if hgmatrix:
    #print(hgmatrix.shape)
    #get the location of the centre of the camera image in the projector image
    dspPoint = virtual_point(frame, hgmatrix)
    print(dspPoint)
    dspPoint = tuple(dspPoint.reshape(1, -1)[0])
    print(dspPoint)
    # draw a circle on where we clicked
    cv2.circle(display, dspPoint[0], 3, (0, 0, 255), thickness=-1, lineType=8)  # color BGR




    # Display the resulting projector image with a dot for the camera location
    cv2.imshow('projector', display)
    if cv2.waitKey(1) & 0xFF == 0x1b:  # ord('q'):
        break
"""
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