EMG.py holds a function EMG that takes in a pdf file, some number k of colors, and 
a flag value that is 0 or 1. The function clusters the colors using multivariate 
gaussian distributions, then replaces each pixel with the center of the cluster
to which the pixel belongs. The result is an image with only k colors. If the flag
value is 0, then no dithering is applied. If the flag is 1, the dithering is applied
which helps the function overcome suboptimal solutions. An example function call is 
given as a comment at the bottom of the file.
