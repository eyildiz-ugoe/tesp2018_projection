import numpy as np
import cv2

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
cv2.imshow('image',display_info(10,50))

k = cv2.waitKey(0)

if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.destroyAllWindows()