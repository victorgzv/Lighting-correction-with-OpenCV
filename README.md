# Lighting-correction-with-OpenCV

- Cleaning an image to obtain a more usable document (Lighting correction) -

-Victor Gonzalez-

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
From this:
![alt text](https://github.com/victorgzv/Lighting-correction-with-OpenCV/blob/master/Text.jpg)
to this more legible and cropped image
![alt text](https://github.com/victorgzv/Lighting-correction-with-OpenCV/blob/master/cleaned.png)


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
