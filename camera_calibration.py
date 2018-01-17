import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

# Bool value to enable plotting
show_plot = True

# Use glob to list required files with a specific pattern in name
images = glob.glob('./CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg')

# Test image to check for undistortion
to_undist = plt.imread(images[0])

# Initialize lists for image and object points
objpoints = []
imagepoints =[]

# Declare sizes
cols, rows = 9, 6

# Prepare object points
objp = np.zeros((cols*rows, 3), np.float32)
objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

# Discover image points in each of the images
for image_name in images:
    image = plt.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Find the intersecting corners in the image
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    # If true value returned, then append object and image points.
    if ret:
        cv2.drawChessboardCorners(image, (cols, rows), corners, ret)
        objpoints.append(objp)
        imagepoints.append(corners)

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imagepoints, gray.shape[::-1], None, None)
# Undistortion check
undist = cv2.undistort(to_undist, mtx, dist, None, mtx)

# plot the distortion vs undistorted image.
if show_plot:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(to_undist)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

# Save the estimated coefficients

print("Saving to file ...")
d = {'mtx':mtx, 'dist':dist}
# Dump camera coefficients to a pickle file as it will be used later.
file = open('camera_coefficients.p', 'wb')
pickle.dump(d, file)
file.close()

print("Saving complete.")