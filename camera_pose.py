import numpy as np
import cv2

# Settings
background_height = 350
background_width = 612
MIN_MATCH_COUNT = 6
MAX_MATCH_COUNT = 20
CAM_WIDTH = 1280
CAM_HEIGHT = 960

# Just using this variable for testing purposes
imageToBeProjected = 'solar_system.jpg'

# marker stuff
marker_file_name = ["marker_one_small.png","marker_two_small.png","marker_three_small.png","marker_four_small.png"]
marker_points = [[0, 0], [0, background_height-100], [background_width-100, 0], [background_width-100, background_height-100]]

def init_webcam(mirror=False):
    cam = []
    camera_height = []
    camera_width = []
    cam = cv2.VideoCapture(1) # try the external camera first
    cam.set(cv2.CAP_PROP_FPS,50)
    cam.set(cv2.CAP_PROP_EXPOSURE,10)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_HEIGHT)
    try:
        ret_val, camera_image = cam.read()
        if len(camera_image) == 0:
            print('Camera not connected')
        camera_height,camera_width,d = camera_image.shape
    except:
        print("No external camera found, attempting to default to  internal camera")
        cam = cv2.VideoCapture(0)
        ret_val, camera_image = cam.read()
        if len(camera_image) == 0:
            raise
        camera_height,camera_width,d = camera_image.shape
    return cam, camera_height, camera_width


# Function to find the homography matrix which transforms from the camera image to the projector image
def get_homography(matches, projectionImage_kp, cameraImage_kp):

    homographyMatrix = []

    # Only perform this if there are enough matches
    if len(matches) > MIN_MATCH_COUNT:
        # Taken from the tutorial
        src_pts = np.float32([projectionImage_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([cameraImage_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Get the homography matrix with RANSAC
        homographyMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))
        matchesMask = None

    return homographyMatrix, matchesMask


def show_matches(projectionImage, cameraImage, projectionImage_kp, cameraImage_kp, homographyMatrix):
    h,w,d = projectionImage.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts,homographyMatrix)
    cameraImage = cv2.polylines(cameraImage, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    visualizationImage = cv2.drawMatches(projectionImage, projectionImage_kp, cameraImage, cameraImage_kp, matches, None, **draw_params)
    cv2.imshow('Debug', visualizationImage)


# Returns the location of the centre of the camera image in the projector image
def virtual_point(homographyMatrix):
    pts = np.float32([ [round(CAM_WIDTH/2),round(CAM_HEIGHT/2)] ]).reshape(-1,1,2)
    m = cv2.invert(homographyMatrix)
    # find the location of this same point in the projector image
    dst = cv2.perspectiveTransform(pts, m[1])
    return dst


if __name__ == '__main__':

    # Set up the camera
    cam, camera_height, camera_width = init_webcam()
    CAM_WIDTH = camera_width
    CAM_HEIGHT = camera_height

    # Load the image that is going to be projected
    projectionImage = cv2.imread(imageToBeProjected)

    # Draw markers on the image
    for marker_index, cp in enumerate(marker_points):
        marker_image = cv2.imread(marker_file_name[marker_index])
        h, w, d = marker_image.shape
        projectionImage[cp[1]:cp[1] + h, cp[0]:cp[0] + w] = marker_image.copy()
    h, w, d = projectionImage.shape

    # initialize the feature detector
    # we use orb, make it ready
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Create an opencv window to display the projection onto
    cv2.namedWindow("Projector", cv2.WINDOW_NORMAL)
    cv2.imshow('Projector', projectionImage)
    cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

    # work with a copy
    processedImage = projectionImage.copy()

    while True:
        orb2 = cv2.ORB_create(nfeatures=500)
        projectionImage_kp = orb2.detect(processedImage, None)
        projectionImage_kp, projectionImage_des = orb2.compute(processedImage, projectionImage_kp)

        # Get an image from the camera
        ret_val, cameraImage = cam.read()

        # Find the keypoints and descriptors with ORB
        cameraImage_kp = orb.detect(cameraImage, None)

        # Compute matches
        matches = []
        if len(cameraImage_kp) > 0:
            cameraImage_kp, cameraImage_des = orb.compute(cameraImage, cameraImage_kp)
            matches = bf.match(projectionImage_des, cameraImage_des)
            if len(matches) > MAX_MATCH_COUNT:
                matches = sorted(matches, key=lambda x: x.distance)[0:MAX_MATCH_COUNT]

        # if we can't find any matches, just keep displaying the image and inform the user
        if len(matches) <= MIN_MATCH_COUNT:
            cv2.imshow('Projector', processedImage)
            print('Could not find matches.')
            cv2.waitKey(10)
            continue

        # Now compute the Homography
        homographyMatrix, matchesMask = get_homography(matches, projectionImage_kp, cameraImage_kp)

        # Visualize!
        show_matches(processedImage, cameraImage, projectionImage_kp, cameraImage_kp, homographyMatrix)

        # Draw a virtual point on the image
        dspPoint = virtual_point(homographyMatrix)
        new_point = dspPoint[0][0]
        cv2.circle(projectionImage, tuple(new_point), 20, (0, 0, 255), thickness=1, lineType=8)  # color BGR

        # Display the resulting projector image with a dot for the camera location
        cv2.imshow('Projector', processedImage)
        if cv2.waitKey(20) == ord('a'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()