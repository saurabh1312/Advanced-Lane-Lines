import numpy as np
import cv2
import pickle
from moviepy.editor import VideoFileClip
from sliding_window import Slider


def abs_sobel_threshold(image, orientation='x', sobel_kernel=3, threshold=(0, 255)):
    # Calculate directional gradient
    abs_sobel = 0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if orientation == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orientation == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)

    # Apply the threshold
    binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
    return binary_output


def color_threshold(image, s_threshold=(0, 255), v_threshold=(0, 255)):
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls_image[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1])] = 1

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_threshold[0]) & (v_channel <= v_threshold[1])] = 1

    binary_output = np.zeros_like(s_channel)
    binary_output[(s_binary == 1) & (v_binary == 1)] = 1
    return binary_output


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


# Read in camera calibration output
dist_pickle = pickle.load(open('./calibration_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


def process_image(img):
    img = cv2.undistort(img, mtx, dist, None, mtx)

    processed_image = np.zeros_like(img[:, :, 0])
    gradx = abs_sobel_threshold(img, orientation='x', threshold=(12, 255))
    grady = abs_sobel_threshold(img, orientation='y', threshold=(25, 255))
    color_binary = color_threshold(img, s_threshold=(100, 255), v_threshold=(50, 255))
    processed_image[((gradx == 1) & (grady == 1) | (color_binary == 1))] = 255

    img_size = (img.shape[1], img.shape[0])
    trap_height_pct = .62
    middle_trap_width_pct = .08
    bottom_trap_width_pct = .76
    avoid_hood_height_pct = .935

    src = np.float32([[img.shape[1] * (.5 - middle_trap_width_pct / 2), img.shape[0] * trap_height_pct],
                      [img.shape[1] * (.5 + middle_trap_width_pct / 2), img.shape[0] * trap_height_pct],
                      [img.shape[1] * (.5 + bottom_trap_width_pct / 2), img.shape[0] * avoid_hood_height_pct],
                      [img.shape[1] * (.5 - bottom_trap_width_pct / 2), img.shape[0] * avoid_hood_height_pct]])
    offset = img_size[0] * .25
    dest = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]],
                       [offset, img_size[1]]])

    m = cv2.getPerspectiveTransform(src, dest)
    m_inv = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(processed_image, m, img_size, flags=cv2.INTER_LINEAR)

    window_width = 25
    window_height = 80

    # Set up the slider
    curve_centers = Slider(width=window_width, height=window_height, pixel_margin=25, ym=10 / 720, xm=4 / 384,
                           smooth=15)
    centroids = curve_centers.find_window_centroids(warped)

    # Points used for drawing left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Points used for finding left and right lanes
    rightx = []
    leftx = []

    # Iterate over each level
    for level in range(0, len(centroids)):
        # Add center value found in frame to lane points
        leftx.append(centroids[level][0])
        rightx.append(centroids[level][1])

        # Draw window areas
        l_mask = window_mask(window_width, window_height, warped, centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, centroids[level][1], level)

        # Add the mask's points here
        l_points[(l_points == 255) | (l_mask == 1)] = 255
        r_points[(r_points == 255) | (r_mask == 1)] = 255

    # Draw the results
    template = np.array(r_points + l_points, np.uint8)  # add left and right pixels
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
    warp = np.array(cv2.merge((warped, warped, warped)), np.uint8)  # make original road pixels 3 color channels

    # fit lane boundaries
    yvals = range(0, warped.shape[0])
    res_yvals = np.arange(warped.shape[0] - (window_height / 2), 0, -window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(
        list(zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis=0),
                 np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(
        list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
                 np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    inner_lane = np.array(
        list(zip(np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
                 np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    # Overlay on the original image
    original_overlaid = cv2.warpPerspective(road, m_inv, img_size, flags=cv2.INTER_LINEAR)
    original_overlaid_bkg = cv2.warpPerspective(road_bkg, m_inv, img_size, flags=cv2.INTER_LINEAR)

    original_overlaid_bkg = cv2.addWeighted(img, 1.0, original_overlaid_bkg, -1.0, 0.0)
    original_overlaid = cv2.addWeighted(original_overlaid_bkg, 1.0, original_overlaid, 0.7, 0.0)

    xm = curve_centers.xm_per_pix
    ym = curve_centers.ym_per_pix

    # Calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * xm
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym, np.array(leftx, np.float32) * xm, 2)
    curverad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * curve_fit_cr[0])

    # Draw text with curvature, offset and speed
    cv2.putText(original_overlaid, 'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(original_overlaid, 'Car is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center',
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return original_overlaid


input_video = 'project_video.mp4'
output_video = './output_videos/project_video.mp4'
# input_video = 'challenge_video.mp4'
# output_video = './output_videos/challenge_video.mp4'
# input_video = 'harder_challenge_video.mp4'
# output_video = './output_videos/harder_challenge_video.mp4'

input_clip = VideoFileClip(input_video)
video_clip = input_clip.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)
