import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def get_calibration_data(glob_arg='../camera_cal/calibration*.jpg'):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(glob_arg)

    img_shape = None

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        img_shape = img.shape[0:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    return mtx, dist

# Define a class to receive the characteristics of each line detection
class LineDetector():
    def __init__(self, calibration_mtx, calibration_dist, margin=100, n_windows=9):

        # Calibration parameters
        self.calibration_mtx = calibration_mtx
        self.calibration_dist = calibration_dist

        # update curvature rad
        self.update_rad_frame = 10
        self.current_frame = 0
        self.left_curvature_rad = None
        self.right_curvature_rad = None

        # Birds eye parameters
        bottom = 680
        middle = 470

        src = np.float32([
            [568, middle],
            [260, bottom],
            [717, middle],
            [1043, bottom]
        ])

        x_start = 200
        x_end = 1000

        dst = np.float32([
            [x_start, 0],
            [x_start, bottom],
            [x_end, 0],
            [x_end, bottom]
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        # params for following video frames
        self.first_frame = True
        self.margin = margin

        # previous fits
        self.left_fit = None
        self.right_fit = None

        # sliding windows for first frame
        self.nwindows = n_windows

        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def undistort_image(self, img):
        # Use cv2.calibrateCamera() and cv2.undistort()
        return cv2.undistort(img, self.calibration_mtx, self.calibration_dist, None, self.calibration_mtx)

    def pipeline(self, undistorted, s_thresh=(170, 255), sx_thresh=(20, 100)):
        undistorted = np.copy(undistorted)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:, :, 1]
        s_channel = hsv[:, :, 2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return color_binary, combined_binary

    def birds_eye_view(self, img):
        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M, img_size,
                                     flags=cv2.INTER_NEAREST)  # keep same size as input image
        return warped

    def get_fits(self, binary_warped):

        # let's do a blind search
        if self.first_frame:

            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            self.nwindows = 9
            # Set height of windows
            window_height = np.int(binary_warped.shape[0] / self.nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(self.nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                # TODO: change to np.median
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.median(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.median(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
            self.first_frame = False
        # just search on the last frames
        else:
            # Assume you now have a new warped binary image
            # from the next frame of video (also called "binary_warped")
            # It's now much easier to find line pixels!
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            left_lane_inds = (
            (nonzerox > (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - self.margin)) & (
            nonzerox < (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + self.margin)))
            right_lane_inds = (
            (nonzerox > (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - self.margin)) & (
            nonzerox < (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + self.margin)))

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

    def update_curvature_rad(self, ploty):
        y_eval = np.max(ploty)

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Calculate the new radii of curvature
        self.left_curvature_rad = ((1 + (2 * self.left_fit[0] * y_eval * ym_per_pix + self.left_fit[1]) ** 2) ** 1.5) / np.absolute(
            2 * self.left_fit[0])
        self.right_curvature_rad = ((1 + (2 * self.right_fit[0] * y_eval * ym_per_pix + self.right_fit[1]) ** 2) ** 1.5) / np.absolute(
            2 * self.right_fit[0])

    def put_prediction(self, img, binary_warped):
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        if self.current_frame == self.update_rad_frame or (self.left_curvature_rad is None and self.right_curvature_rad is None):
            self.update_curvature_rad(ploty)
            self.current_frame = 0
        else:
            self.current_frame += 1

        result = cv2.putText(result, "left curve rad: %f.2 m" % self.left_curvature_rad, (50, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                             (0, 255, 0), 2, cv2.LINE_AA)
        result = cv2.putText(result, "right curve rad: %f.2 m" % self.right_curvature_rad, (50, 75),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                             (0, 255, 0), 2, cv2.LINE_AA)

        return result

    def image_pipeline(self, image):
        self.first_frame = True
        undist = self.undistort_image(image)
        _, combined_binary = self.pipeline(undist)
        binary_warped = self.birds_eye_view(combined_binary)
        self.get_fits(binary_warped)
        output = self.put_prediction(undist, binary_warped)
        return output

    def produce_video(self, input_path, output_path):
        clip2 = VideoFileClip(input_path)
        clip = clip2.fl_image(self.image_pipeline)
        clip.write_videofile(output_path, audio=False)

mtx, dist = get_calibration_data()
ld = LineDetector(mtx, dist)
ld.produce_video('../project_video.mp4', '../output_images/output.mp4')
