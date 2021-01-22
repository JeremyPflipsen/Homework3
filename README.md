EMG.py holds a function EMG that takes in a jpg file, some number k of colors, and 
a flag value that is 0 or 1. The function clusters the colors of the pixels using k multivariate 
gaussian distributions, then replaces each pixel with the center of the cluster
to which the pixel belongs. The result is the original image but with only k colors. If the flag
value is 0, then no dithering is applied. If the flag is 1, dithering is applied
which helps the function overcome suboptimal solutions. An example function call is 
given as a comment at the bottom of the file EDM file.
