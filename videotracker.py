import numpy as np
import cv2
import pickle
import glob
from tracker import tracker
from moviepy.editor import VideoFileClip

dist_pickle = pickle.load(open("camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def color_threshold(image, sthresh = (0, 255), vthresh= (0, 255)):
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

	l_channel = hls[:,:,1]
	l_binary = np.zeros_like(s_channel)
	l_binary[(l_channel > sthresh[0]) & (l_channel <= sthresh[1])] = 1

	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	v_channel = hsv[:,:,1]
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

	output = np.zeros_like(s_channel)
	output[((s_binary == 1) & (v_binary == 1) & (l_binary == 1))] = 1
	return output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

polygon_points_old = None

def process(img):
	global polygon_points_old
	preprocessImage = np.zeros_like(img[:,:,0])
	gradx = abs_sobel_thresh(img, orient = 'x', thresh_min=8, thresh_max=100)
	grady = abs_sobel_thresh(img, orient = 'y', thresh_min=30, thresh_max=100)
	mag_binary = mag_thresh(img, sobel_kernel= 3, mag_thresh=(50, 255))
	c_binary = color_threshold(img, sthresh = (100, 255), vthresh = (60, 255))
	preprocessImage[((gradx == 1) & (grady == 1))] = 255
	preprocessImage[((mag_binary == 1))] = 255
	HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))
	sensitivity_1 = 68
	white = cv2.inRange(HSV, (0,0,255-sensitivity_1), (255,20,255))
	sensitivity_2 = 60
	HSL = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	white_2 = cv2.inRange(HSL, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
	white_3 = cv2.inRange(img, (180,180,180), (255,255,255))
	preprocessImage[((yellow == 1) | (white == 1) | (white_2 == 1) | (white_3 == 1) | (c_binary == 1))] = 255

	img_size = (img.shape[1], img.shape[0])
	src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) + 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 80), img_size[1] / 2 + 100]])
	dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
	
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(preprocessImage, M, img_size, flags = cv2.INTER_LINEAR)
	
	window_width = 40
	window_height = 60

	curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 25, My_ym = 30/720, My_xm = 3.7/700, Mysmooth_factor = 10)

	window_centroids = curve_centers.find_window_centroids(warped)

	l_points = np.zeros_like(warped)
	r_points = np.zeros_like(warped)

	leftx = []
	rightx = []

	for level in range(0, len(window_centroids)):

		leftx.append(window_centroids[level][0])
		rightx.append(window_centroids[level][1])
		l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0],level)
		r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1],level)

		l_points[(l_points == 255) | (l_mask == 1)] = 255
		r_points[(r_points == 255) | (r_mask == 1)] = 255
 	
	#draw results
	template = np.array(r_points+l_points, np.uint8)
	zero_channel = np.zeros_like(template)
	template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
	warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
	result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

	yvals = range(0, warped.shape[0])

	res_yvals = np.arange(warped.shape[0] - (window_height/2), 0, -window_height)

	left_fit = np.polyfit(res_yvals, leftx, 2)
	left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
	left_fitx = np.array(left_fitx, np.int32)

	right_fit = np.polyfit(res_yvals, rightx, 2)
	right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
	right_fitx = np.array(right_fitx, np.int32)

	left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2, left_fitx[::-1]+window_width/2), axis = 0), np.concatenate((yvals,yvals[::-1]), axis = 0))), np.int32)
	right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2, right_fitx[::-1]+window_width/2), axis = 0), np.concatenate((yvals,yvals[::-1]), axis = 0))), np.int32)
	inner_lane = np.array(list(zip(np.concatenate((left_fitx + window_width/2, right_fitx[::-1]+window_width/2), axis = 0), np.concatenate((yvals,yvals[::-1]), axis = 0))), np.int32)

	road = np.zeros_like(img)
	road_bkg = np.zeros_like(img)

	if (polygon_points_old == None):
		polygon_points_old = np.float32([[left_lane], [right_lane], [inner_lane]])

	a = polygon_points_old
	b = np.float32([[left_lane], [right_lane], [inner_lane]])
	ret1 = cv2.matchShapes(np.asarray(a[0]),np.asarray(b[0]),1,0.0)
	ret2 = cv2.matchShapes(np.asarray(a[1]),np.asarray(b[1]),1,0.0)
	ret3 = cv2.matchShapes(np.asarray(a[2]),np.asarray(b[2]),1,0.0)

	if ((ret1 < 0.50) & (ret2 < 0.50) & (ret3 < 0.50)):
		cv2.fillPoly(road, [left_lane], color = [0,255,0])
		cv2.fillPoly(road, [right_lane], color = [0,255,0])
		cv2.fillPoly(road, [inner_lane], color = [0,128,0])
		cv2.fillPoly(road_bkg, [left_lane], color = [255,255,255])
		cv2.fillPoly(road_bkg, [right_lane], color = [255,255,255])
		polygon_points_old = [[left_lane], [right_lane], [inner_lane]]

	else:
		cv2.fillPoly(road, polygon_points_old[0], color = [0,255,0])
		cv2.fillPoly(road, polygon_points_old[1], color = [0,255,0])
		cv2.fillPoly(road, polygon_points_old[2], color = [0,128,0])
		cv2.fillPoly(road_bkg, polygon_points_old[0], color = [255,255,255])
		cv2.fillPoly(road_bkg, polygon_points_old[1], color = [255,255,255])	

	road_warped = cv2.warpPerspective(road, Minv, img_size, flags = cv2.INTER_LINEAR)
	road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags = cv2.INTER_LINEAR)
	base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
	result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)

	ym_per_pix = curve_centers.ym_per_pix
	xm_per_pix = curve_centers.xm_per_pix

	curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx, np.float32)*xm_per_pix,2)
	curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])
	camera_center = (left_fitx[-1] + right_fitx[-1])/2
	center_diff = (camera_center - warped.shape[1]/2)*xm_per_pix
	side_pos = 'left'
	if center_diff <=0:
		side_pos = 'right'

	cv2.putText(result, 'Radius of Curvature = '+str(np.round(curverad, 3))+ '(m)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
	cv2.putText(result, 'Vehicle is '+ str(abs(np.round(center_diff, 3)))+ 'm '+side_pos+' of center', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
	return result

outputVideo = 'output_tracked.mp4'
inputVideo = 'project_video.mp4'



clip1 = VideoFileClip(inputVideo)
video_clip = clip1.fl_image(process)
video_clip.write_videofile(outputVideo, audio = False)


