import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test.jpg')

y_size = image.shape[0]
x_size = image.shape[1]
color_select = np.copy(image)
line_image = np.copy(image)

print(x_size, y_size)
### Parameters
## Color
red_threshold = 192
green_threshold = 128
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

## ROI
bottom = 50
left_bottom = [0 + bottom, y_size]
right_bottom = [x_size - bottom, y_size - 1]
apex = [470, 320]


### Procedure
## Color
color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
                   (image[:, :, 1] < rgb_threshold[1]) | \
                   (image[:, :, 2] < rgb_threshold[2])

# Mask color selection
color_select[color_thresholds] = [0, 0, 0]

## Get ROI
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

XX, YY = np.meshgrid(np.arange(0, x_size), np.arange(0, y_size))
region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                    (YY > (XX * fit_right[0] + fit_right[1])) & \
                    (YY < (XX * fit_bottom[0] + fit_bottom[1]))

# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# Display our two output images
plt.imshow(color_select)
plt.show()
plt.imshow(line_image)
plt.show()
