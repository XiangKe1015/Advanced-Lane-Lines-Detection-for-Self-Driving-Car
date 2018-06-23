import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

def calibrata_camera(cal_images,nx=9,ny=6):
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((nx*ny, 3),np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane

	# Make a list of calibration images 
	# Step through the list and search for chessboard corners
	for index, image in enumerate(cal_images):
		image = cv2.imread(image)
		img_size=image.shape[1::-1]
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#Find the chessboard points and image points
		ret, corners = cv2.findChessboardCorners(gray,(9,6),None)
		#If found, add object points and image points
		if ret==True:
			objpoints.append(objp)
			imgpoints.append(corners)
			#Draw and display corners
			cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
			write_name = 'corners_found/'+ str(index)+ '.jpg'
			cv2.imwrite(write_name,image)
			#cv2.imshow('image',image)
			#cv2.waitKey(500)
	#cv2.destroyAllWindows()
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
	# Save the camera calibration result for later
	dist_pickle = {}
	dist_pickle["mtx"] = mtx
	dist_pickle["dist"] = dist
	dist_pickle["objpoints"]=objpoints
	dist_pickle["imgpoints"]=imgpoints
	pickle.dump( dist_pickle, open( "camera_cal/dist_pickle.p", "wb" ) )
	

def cal_undistort(image,cal_file='camera_cal/dist_pickle.p'):
    # Use cv2.calibrateCamera() and cv2.undistort()
    dist_pickle = pickle.load( open( cal_file, "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    undist = cv2.undistort(image,mtx,dist,None,mtx)  

    return undist

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx=np.absolute(sobelx)
    abs_sobely=np.absolute(sobely)
    scaled_sobelx=np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaled_sobely=np.uint8(255*abs_sobely/np.max(abs_sobely))
    if orient=='x':
        grad_binary=np.zeros_like(scaled_sobelx)
        grad_binary[(scaled_sobelx>=thresh[0])&(scaled_sobelx<=thresh[1])]=1
    
    elif orient=='y':
        grad_binary=np.zeros_like(scaled_sobely)
        grad_binary[(scaled_sobely>=thresh[0])&(scaled_sobely<=thresh[1])]=1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0,255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag=np.sqrt(sobelx**2+sobely**2)
    scaled_sobel=np.uint8(255*mag/np.max(mag))
    mag_binary=np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel>=mag_thresh[0])&(scaled_sobel<=mag_thresh[1])]=1
    return mag_binary

def dir_threshold (image,sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dir_sobel=np.arctan2(np.absolute(sobely),np.absolute(sobelx))
    dir_binary=np.zeros_like(dir_sobel)
    dir_binary[(dir_sobel>=thresh[0])&(dir_sobel<=thresh[1])]=1
    return dir_binary

# Combining Thresholds
def combined_gradient(image):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 100)) 
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100)) 
    mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.6, 1.0))
    
    combined_binary = np.zeros_like(dir_binary)
    combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined_binary

def hls_hsv_select(img, sthresh=(0, 255), vthresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    sbinary_output = np.zeros_like(s_channel)
    sbinary_output[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel=hsv[:,:,2]
    vbinary_output = np.zeros_like(v_channel)
    vbinary_output[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(sbinary_output == 1) & (vbinary_output == 1)] = 1
    
    return output


def threshold(image):
    
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    gradient_binary = combined_gradient(image)

    s_binary=hls_hsv_select(image, sthresh=(100, 255), vthresh=(200,255)) # sthresh=(200, 255), vthresh=(200,255)

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(gradient_binary), gradient_binary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(gradient_binary)
    combined_binary[(s_binary == 1) | (gradient_binary == 1)] = 1
    
    return combined_binary

def corners_unwarp(img):
    img_size=img.shape[1::-1]
    #src=np.float32([[537.6,  475.2], [742.4,  475.2], [1126.4,  673.2], [153.6,  673.2]]) 
    #dst=np.float32([[422.4,   0.], [857.6,   0.],[857.6, 720.], [422.4, 720.]])
    src=np.float32([[200,715],[1150,715],[620,450],[725,450]]) 
    dst=np.float32([[280,715], [950,715],[280,0], [950,0]])
    M=cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped=cv2.warpPerspective(img,M,img_size, flags=cv2.INTER_LINEAR)
    
    return  warped, Minv

def find_lanelines(binary_warped):

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    nwindows = 9
    window_height = np.int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 100
    minpix = 50
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    '''out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig('lane.jpg')
    plt.show()'''
    return ploty, left_fitx, right_fitx

def find_curvature(binary_warped):

    ploty, left_fitx, right_fitx=find_lanelines(binary_warped)

    '''ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])# to cover same y-range as image
    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
	# For each y position generate random x position within +/-50 pix
	# of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([left_fitx[-1] + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                              for y in ploty])
    rightx = np.array([right_fitx[-1] + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                for y in ploty])
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

	# Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Plot up the fake data
    mark_size = 3
    plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
    plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images
    plt.show()'''

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    '''y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48'''

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    img_middle = (left_fitx[-1] + right_fitx[-1])//2
    veh_pos = binary_warped.shape[1]//2
    dx = (veh_pos - img_middle)*xm_per_pix # Positive if on right, Negative on left

    return left_curverad, right_curverad, dx

def vis(undist,warped,Minv):
	# Create an image to draw the lines on
	ploty, left_fitx, right_fitx=find_lanelines(warped)

	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	#plt.imshow(result)
	#plt.show()
	
	return result

def put_text(image, left_curverad,right_curverad, dx):
    # add text backdrop
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image,'Left radius of curvature  = %.2f m'%(left_curverad),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,'Right radius of curvature = %.2f m'%(right_curverad),(50,80), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,'Vehicle position : %.2f m %s of center'%(abs(dx), 'left' if dx < 0 else 'right'),(50,110),
                        font, 1,(255,255,255),2,cv2.LINE_AA)
    return image