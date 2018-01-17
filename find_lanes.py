import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import glob
from moviepy.editor import VideoFileClip

# Function to undistort the image with distortion matrix.
def undistortImage(img, mtx, dist):

    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

# Function that applies given thresholds to sobel.
# Can be called for both sobel x and y separately.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel_grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    if orient == 'y':
        sobel_grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_abs = np.absolute(sobel_grad)

    sobel_scaled = np.uint8(255 * sobel_abs/np.max(sobel_abs))

    sobel_threshold = np.zeros_like(sobel_scaled)
    sobel_threshold[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1

    return sobel_threshold

# Function to apply given thresholds to sobel magnitude.
def mag_sobel_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_mag = np.hypot(sobelx, sobely)

    sobel_scaled = np.uint8(255 * sobel_mag/np.max(sobel_mag))

    sobel_thresh = np.zeros_like(sobel_scaled)
    sobel_thresh[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1

    return sobel_thresh

# Function to apply given thresholds to sobel gradient.
def gradient_sobel_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_grad = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    sobel_thresh = np.zeros_like(sobel_grad)
    sobel_thresh[(sobel_grad >= thresh[0]) & (sobel_grad <= thresh[1])] = 1

    return sobel_thresh

# This function creates the binary image.
# This function takes the input of undistorted image.
# All thresholds are defined here and other respective functions are called.
# Finally the union of all processing is returned as binary image.
def createBinaryThresholded(image_undist):
    #Sobel x parameters
    sobel_x_kernel = 5
    sobel_x_thresh = (20, 80)

    # Sobel magnitude parameters
    sobel_mag_kernel = 15
    sobel_mag_thresh = (30, 130)

    # Sobel gradient parameters
    sobel_grad_kernel = 15
    sobel_grad_thresh = (np.deg2rad(40), np.deg2rad(75))

    # Saturation channel thresholds
    # Note: The image has to be converted to HSL colorspace first
    s_thresh = (170, 255)
    s_thresh_2 = (150, 255)

    # Hue channel thresholds.
    # Note: The image has to be converted to HSL colorspace first
    h_thresh = (20, 30)

    # Red channel thresholds in RGB colorspace.
    r_channel_thresh = (220, 255)

    # Apply the defined parameters to
    sobel_x = abs_sobel_thresh(image_undist, 'x', sobel_x_kernel, sobel_x_thresh)
    sobel_mag = mag_sobel_thresh(image_undist, sobel_mag_kernel, sobel_mag_thresh)
    sobel_grad = gradient_sobel_thresh(image_undist, sobel_grad_kernel, sobel_grad_thresh)

    # HLS transformation
    image_hsl = cv2.cvtColor(image_undist, cv2.COLOR_RGB2HLS)
    image_h = image_hsl[:,:,0]
    image_s = image_hsl[:,:,2]

    # Saturation channel thresholding
    s_binary = np.zeros_like(image_s)
    s_binary[(image_s >= s_thresh[0]) & (image_s <= s_thresh[1])] = 1
    s_binary_2 = np.zeros_like(image_s)
    s_binary_2[(image_s >= s_thresh_2[0]) & (image_s <= s_thresh_2[1])] = 1

    # Hue channel thresholding
    h_binary = np.zeros_like(image_h)
    h_binary[(image_h >= h_thresh[0]) & (image_h <= h_thresh[1])] = 1

    # Red channel thresholding
    r_channel = image_undist[:,:,0]

    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_channel_thresh[0]) & (r_channel <= r_channel_thresh[1])] = 1

    # Combined thresholding
    binary_image = np.zeros_like(sobel_x)
    binary_image[(((sobel_x == 1) & (sobel_mag == 1)) & ((sobel_x == 1) & (sobel_grad == 1))) |
             (((s_binary == 1) & (r_binary == 1)) | ((h_binary == 1) & (s_binary_2 == 1)))] = 1

    return binary_image

# Function that performs the sliding window algorithm for lane finding.
# The inputs of this function are as follows,
# nonzeros - The non zero indexes of the binary image
# n - number of windows (in height)
# m - left and right margin for each window, in pixels
# h - height of the window, in pixels
# qlf - minimum number of inliers inside a window for realigning.
# left_pos - The starting position for the left lane
# right_pos - The starting position for the right lane
# out_img - The output image where the sliding windows are drawn on to.

# This function outputs the qualified left and right indexes of lanes and output image.
def slidingWindowSearch(nonzeros, n, m, h, qlf, left_pos, right_pos, out_img):
    # Initialize empty lists for holding valid left and right lanes
    lefts = []
    rights = []

    # Find the nonzero pixel indexes in x and y direction
    nonzeros_x = nonzeros[1]
    nonzeros_y = nonzeros[0]

    # Perform sliding window search
    for window in range(n):
        # Define the bases for the window
        bottom_base = out_img.shape[0] - (window * h)
        top_base = bottom_base - h

        # Left positions
        left_low = left_pos - m
        left_high = left_pos + m

        # Right positions
        right_low = right_pos - m
        right_high = right_pos + m

        # Draw the current window box to the image
        cv2.rectangle(out_img, (left_low, bottom_base), (left_high, top_base), (0, 255, 0), 2)
        cv2.rectangle(out_img, (right_low, bottom_base), (right_high, top_base), (0, 255, 0), 2)

        # Find the indexes of inliers inside the window
        left_lane_idx = ((nonzeros_x >= left_low) & (nonzeros_x <= left_high) &
                        (nonzeros_y <= bottom_base) & (nonzeros_y >= top_base)).nonzero()[0]
        right_lane_idx = ((nonzeros_x >= right_low) & (nonzeros_x <= right_high) &
                         (nonzeros_y <= bottom_base) & (nonzeros_y >= top_base)).nonzero()[0]

        # Append the indexes
        lefts.append(left_lane_idx)
        rights.append(right_lane_idx)

        # Realign the box if number of detected pixels are more than define qualified.
        if np.sum(left_lane_idx) >= qlf:
            left_pos = np.int(np.mean(nonzeros_x[left_lane_idx]))

        if np.sum(right_lane_idx) >= qlf:
            right_pos = np.int(np.mean(nonzeros_x[right_lane_idx]))

    lefts = np.array(lefts)
    rights = np.array(rights)

    lefts = np.concatenate(lefts)
    rights = np.concatenate(rights)

    return lefts, rights, out_img

# Function to compute the radius of curvature
# Takes the input of x-x pixels, y-y pixels and width of lane in pixels.
def computeRadiusOfCurvature(x, y, pix_lane_w):
    # Define pixel scaling factors in x and y for real world space.
    y_pix_to_m = 40/720
    x_pix_to_m = 3.7/(pix_lane_w + 24) # 24 pixels added for tuning

    # Define a point in y direction to calculate the radius of curvature
    y_max_cur = image.shape[0]

    # Calculate the fit parameters in real world.
    fit_real_world = np.polyfit(y * y_pix_to_m, x * x_pix_to_m, 2)

    # Compute the radius of curvature.
    curvature = ((1 + ((2 * fit_real_world[0] * y_max_cur * y_pix_to_m) + fit_real_world[1])**2)**1.5)\
                     /np.absolute(2 * fit_real_world[0])

    return curvature

# Function to determine the position of the lane line.
def computeLaneOffset(fit, y):
    offset = fit[0] * y ** 2 + fit[1] * y + fit[2]

    return offset

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Fit parameters list. Contains all valid fit parameters
        self.fitparam = []
        # Contains the best fit calculated from a subset of fit parameters
        self.bestfit = np.array([0., 0., 0.])
        # Contains the computed radius of curvatures as a list
        self.radiusofcurvature = []
        # Contains the lane offset from the center mounter camera.
        self.laneoffset = []

#######################################################################################################################

# Get calibration data
calibration_file = 'camera_coefficients.p'

with open(calibration_file, mode='rb') as f:
    calibration = pickle.load(f)

mtx, dist = calibration['mtx'], calibration['dist']

# Object definitions for the left and right lane.
ob_left_line = Line()
ob_right_line = Line()

reset_counter = 0

def findLaneLines(image):
    # Undistort the test image
    image_undist = undistortImage(image, mtx, dist)

    # Get the binary thresholded image
    binary_image = createBinaryThresholded(image_undist)

    # Prepare source and destination points for perspective transformation
    img_shape = image.shape
    srcpts = np.float32([[275, 677], [600, 446], [685, 446], [1045, 677]])
    dstpts = np.float32([[426, img_shape[0]], [426, 0], [852, 0], [852, img_shape[0]]])

    # Calculate the transformation matrix
    M = cv2.getPerspectiveTransform(srcpts, dstpts)

    # Calculate the inverse transformation matrix for inverse transformation.
    Minv = cv2.getPerspectiveTransform(dstpts, srcpts)

    # Warp the image with the perspective transformation matrix
    warped = cv2.warpPerspective(binary_image, M, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR)

    ## Sliding window method
    histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)

    out_img = np.dstack((warped, warped, warped)) * 255

    # Find the argmax of found lane.
    max_lane = np.argmax(histogram)
    midppoint = warped.shape[1]//2

    pixel_lane_width = np.int(np.absolute(dstpts[0, 0] - dstpts[3, 0]))

    if max_lane <= midppoint:
        left_max = max_lane
        right_max = left_max + pixel_lane_width
    else:
        right_max = max_lane
        left_max = right_max - pixel_lane_width

    nonzeros = warped.nonzero()
    nonzeros_x = nonzeros[1]
    nonzeros_y = nonzeros[0]

    nwindows = 9
    window_margin = 60
    window_height = warped.shape[0]//nwindows
    minimum_qualify = 30 # 50. Set to 30 since the window size is small.

    # Perform the sliding window search
    if ob_left_line.detected == False & ob_right_line.detected == False:
        lefts, rights, out_img = slidingWindowSearch(nonzeros, nwindows, window_margin, window_height,
                                                     minimum_qualify, left_max, right_max, out_img)
    else:
        fit_l = ob_left_line.bestfit

        fit_r = ob_right_line.bestfit

        lefts = (nonzeros_x >= ((fit_l[0] * nonzeros_y ** 2) + fit_l[1] * nonzeros_y + fit_l[2] - window_margin*0.5)) & \
                (nonzeros_x <= ((fit_l[0] * nonzeros_y ** 2) + fit_l[1] * nonzeros_y + fit_l[2] + window_margin*0.5))

        rights = (nonzeros_x >= ((fit_r[0] * nonzeros_y ** 2) + fit_r[1] * nonzeros_y + fit_r[2] - window_margin*0.5)) & \
                (nonzeros_x <= ((fit_r[0] * nonzeros_y ** 2) + fit_r[1] * nonzeros_y + fit_r[2] + window_margin*0.5))

    # Prepare output image
    out_img[nonzeros_y, nonzeros_x] = [0, 0, 0]
    out_img[nonzeros_y[lefts], nonzeros_x[lefts]] = [255, 0, 0]
    out_img[nonzeros_y[rights], nonzeros_x[rights]] = [0, 0, 255]

    # Polyfit
    leftx = nonzeros_x[lefts]
    lefty = nonzeros_y[lefts]
    rightx = nonzeros_x[rights]
    righty = nonzeros_y[rights]

    fit_valid_left = False
    fit_valid_right = False

    if (leftx.size != 0) & (lefty.size != 0):
        fit_left = np.polyfit(lefty, leftx, 2, full=True)
        fit_valid_left = True

    if (rightx.size != 0) & (righty.size != 0):
        fit_right = np.polyfit(righty, rightx, 2, full=True)
        fit_valid_right = True

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    if fit_valid_left & fit_valid_right:
        y_spaced = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        xleft_fit = fit_left[0][0] * y_spaced ** 2 + fit_left[0][1] * y_spaced + fit_left[0][2]
        xright_fit = fit_right[0][0] * y_spaced ** 2 + fit_right[0][1] * y_spaced + fit_right[0][2]

        difference_in_x = np.mean(np.absolute((xleft_fit - xright_fit)))

        curvature_left = computeRadiusOfCurvature(leftx, lefty, pixel_lane_width)
        curvature_right = computeRadiusOfCurvature(rightx, righty, pixel_lane_width)

        ob_left_line.radiusofcurvature.append(curvature_left)

        ob_right_line.radiusofcurvature.append(curvature_right)

        if (np.absolute(difference_in_x - 426) < 50):
            ob_left_line.detected = True
            ob_right_line.detected = True

            ob_left_line.fitparam.append(fit_left[0])
            ob_right_line.fitparam.append(fit_right[0])

            leftparam = np.array(ob_left_line.fitparam)
            rightparam = np.array(ob_right_line.fitparam)

            current_length_left = 0
            if (leftparam.shape[0] > 0):
                current_length_left = leftparam.shape[0]

            current_length_right = 0
            if (rightparam.shape[0] > 0):
                current_length_right = rightparam.shape[0]

            keep_hist = 5 #5

            wanted_length_left = np.min([current_length_left, keep_hist])
            wanted_length_right = np.min([current_length_right, keep_hist])

            ob_left_line.bestfit = np.mean(leftparam[-wanted_length_left:], axis=0)
            ob_right_line.bestfit = np.mean(rightparam[-wanted_length_right:], axis=0)

            global reset_counter
            reset_counter = 0
        else:
            # global reset_counter
            reset_counter += 1

        if reset_counter > 25:
            ob_left_line.detected = False
            ob_right_line.detected = False

            # global reset_counter
            reset_counter = 0

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([xleft_fit, y_spaced]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([xright_fit, y_spaced])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    unwarped_lane = cv2.warpPerspective(color_warp, Minv, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR)
    unwarped_detected_lane = cv2.warpPerspective(out_img, Minv, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR)

    unwarped_full = cv2.addWeighted(image_undist, 1, unwarped_lane, 0.4, 0)
    unwarped_full = cv2.addWeighted(unwarped_full, 1, unwarped_detected_lane, 0.7, 0)

    ## Calculate the radius of curvature in real world space
    y_max_cur = image.shape[0]

    left_offset = computeLaneOffset(ob_left_line.bestfit, y_max_cur)
    right_offset = computeLaneOffset(ob_right_line.bestfit, y_max_cur)

    ob_left_line.laneoffset.append(left_offset)
    ob_right_line.laneoffset.append(right_offset)

    # write text on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_of_curvature = (ob_left_line.radiusofcurvature[-1] + ob_right_line.radiusofcurvature[-1])/2
    radius_of_curvature = format(radius_of_curvature, '.2f')
    text = "Radius of curvature " + str(radius_of_curvature) + "m"
    cv2.putText(unwarped_full, text, (400, 100), font, 1, (255, 255, 255), 2)

    # Lane offset
    left_offset -= (1280 / 2)
    left_offset *= 3.7/(426. + 24.)
    right_offset -= (1280 / 2)
    right_offset *= 3.7/(426. + 24.)

    half_lane_width = 3.7/2

    if (np.absolute(left_offset) - half_lane_width) > 0:
        lane_offset = np.absolute(left_offset) - half_lane_width
        lane_offset = format(lane_offset, '.2f')
        lane_text = "Ego center is " + str(lane_offset) + " m left of lane center"
    elif (np.absolute(right_offset) - half_lane_width) > 0:
        lane_offset = np.absolute(right_offset) - half_lane_width
        lane_offset = format(lane_offset, '.2f')
        lane_text = "Ego center is " + str(lane_offset) + " m right of lane center"
    else:
        lane_text = "Ego center is at lane center"

    cv2.putText(unwarped_full, lane_text, (400, 150), font, 1, (255, 255, 255), 2)

    return unwarped_full

# Get list of images
test_images = glob.glob('./CarND-Advanced-Lane-Lines/test_images/*.jpg')
image = plt.imread(test_images[3]) # 1, 2

print("Image shape : ", image.shape)

white_output = 'project_video_output.mp4'
clip1 = VideoFileClip("CarND-Advanced-Lane-Lines/project_video.mp4")

white_clip = clip1.fl_image(findLaneLines)
white_clip.write_videofile(white_output, audio=False)

d = {'ob_left_line':ob_left_line, 'ob_right_line':ob_right_line}
# Dump lane objects to a pickle file as it will be used later.
file = open('laneobjects.p', 'wb')
pickle.dump(d, file)
file.close()