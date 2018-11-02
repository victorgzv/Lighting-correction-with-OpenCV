'''
- Clean the image to obtain a more usable document (Lighting correction) -

Victor Gonzalez - D16123580 - DT228

The human eye like contrast in images. Bright regions should be really bright and dark should be really dark. To achieve this in this assignment I have used histogram equalisation.
Histogram equalisation (equalizeHist) takes the values that are very low in the original picture and strech them out to range from 0 to 255.
This way the new image is using all the available values and will have a better contrast.

By applying histogram equalisation directly on the channels of color images (RGB) I didn't obtain better results.
Therefore, I coverted the color space of the image into YUV which separates the intensity value from the color components.
I have equalised the Y channel and converted back to RGB. This makes the picture have a much better contrast and doesn't disturb the color of the image.

However, I came across CLAHE (Contrast Limited Adaptive Histogram Equalization)
This method gives better results than applying histogram equalization on the Y channel.
CLAHE divides the image into small blocks 8x8 which are equalised as usual using equalizeHist. I tested this method using different values for the clipLimit variable. 2.5 is the value that work best for the image.
As a resut of this method I have obtained a more usuable document and I have chosen this way over equalising the intensity channel of the image.

The resulting image is converted to the grey color space to apply futher methods.
I have used Adaptive Thresholding for masking the image. This method takes region by region and classifies pixels in black and white.
This method provides changes in illumination and the contrast of the image is improved. I used Gaussian thresholding which removes the right bottom part of the image.
The text is now more legible and allows me to proceed to cropping out the desired region of text.

For the second part of this assignment I created a function that crops out the region of text automatically.
I used the canny edge detection to find all the areas of text.
The dilation method makes all the edges merge together into one big white area that covers all the text. This is the area of text I am interested to isolate.
I also used the contours method to find the borders of the white area. This variables (width and height) were used to crop the image obatained after applying Clahe.
Finally, the new image is displayed to the user beside the original one.

This script has been tested with the given Text.jgp and similar images found on the internet. It successfully work and isolates the text in similar images.

References:

    - Histogram equalisation of RGB Images:
        https://prateekvjoshi.com/2013/11/22/histogram-equalization-of-rgb-images/?fbclid=IwAR2EigPD-DYdAAbZJhZKnqlWrlpcsaqrvhS_1c60wJOu5KwPwKI7x3aaYRE
    - CLAHE
        Understanding different algorithms for histogram equalization:
        https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    - Canny Edge detection
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
    - Detecting text region in images
        I learned how to apply canny, dilation and contours applied to the segmentation lecture that was explained in class.
        https://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html
    - OpenCV documentation
        I refer to the actual opencv documentation to understand every method used in this assignment
        https://docs.opencv.org/3.1.0/
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
