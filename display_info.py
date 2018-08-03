import cv2

refPt = []

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
	global coordinates

    # if the left mouse button was clicked, record the starting
	if event == cv2.EVENT_LBUTTONDBLCLK:
        coordinates = [(x, y)]

    # draw a circle
    cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
    cv2.imshow("image", img)

def display_info(coordinate_x, coordinate_y):

    # properties of the text to be used on the display window
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (coordinate_x + 10, coordinate_y + 10) # next to the object
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 1

    cv2.putText(img, 'Planet Info: Tralala!',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return img


img = cv2.imread('solar_system.jpg',1)
clone = img.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
cv2.imshow('image',display_info(10,50))

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", img)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		img = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()