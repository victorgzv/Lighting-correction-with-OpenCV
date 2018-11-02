'''
- Clean the image to obtain a more usable document (Lighting correction) -

Victor Gonzalez - D16123580 - DT228

'''
# import the necessary packages:
import cv2
import numpy as np
import easygui
from matplotlib import pyplot as plt
from matplotlib import image as image

# Opening an image using a File Open dialog:
#f = easygui.fileopenbox()
#Opening an image
I = cv2.imread("Text.jpg")

#The following code extracts the Y channel, apply the histm equalization and convert it back to RGB
'''
YUV= cv2.cvtColor(I,cv2.COLOR_BGR2YUV)
YUV[:,:,0]=cv2.equalizeHist(YUV[:,:,0])

# convert the YUV image back to RGB format
img_output = cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR)
'''

RGB = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)#convert to RGB
R,G,B = cv2.split(RGB)

#Create a CLAHE object: The image is divided into small block 8x8 which they are equalized as usual.
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
#Applying this method to each channel of the color image
output_2R = clahe.apply(R)
output_2G = clahe.apply(G)
output_2B = clahe.apply(B)

#mergin each channel back to one
img_output = cv2.merge((output_2R,output_2G,output_2B))
#coverting image from RGB to Grayscale
eq=cv2.cvtColor(img_output,cv2.COLOR_BGR2GRAY)
#Using image thresholding to classify pixels as dark or light
#This method provides changes in illumination and the contrast of the image is improved.
gauss = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 45)

def cropText(img,clear_image):
    edges = cv2.Canny(img,100,200)#canny edge detection
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))  # Define the area of focus around each pixel
    dilation = cv2.dilate(edges, kernel, iterations=9)  # dilate merge all edges into one big group
    _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # This method returns 3 variables for getting contours
    for contour in contours:
            # get sizes of the rectangle return by the contours
            [x, y, w, h] = cv2.boundingRect(contour)
            #cropping image from the best contrast image obtained
            cropped = clear_image[y :y +  h , x : x + w]

    #Displaying images on screen outside of foor loop.
    cv2.imshow("original" , I)
    cv2.imshow("cropped" , cropped)
    cv2.imshow("equalised",eq)
    cv2.imshow("Gaussian",gauss)
    cv2.imshow("Edges",edges)
    key = cv2.waitKey(0)


#function call
cropText(gauss,eq)

'''
#Ploting the original image beside its histogram allows me to see the differences between the orginal image and the equalised image.
plt.figure()
plt.subplot(2,2,1) #there will be 2 rows, 2 columns of plots, we will plot this first one in location 1
plt.imshow(I)

grey = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
Values = grey.ravel()#unravelled from matrix form to a 1D
plt.subplot(2,2,2)#location 2
plt.hist(Values,bins=256,range=[0,256]);

H = cv2.equalizeHist(eq)
plt.subplot(2,2,3)#location 3
plt.imshow(H, cmap='gray')

G= H.ravel()
plt.subplot(2,2,4)#location 4
plt.hist(G,bins=256,range=[0,256]);
plt.show()
'''
