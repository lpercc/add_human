import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import json
import random
import argparse

## add human to the env images

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return
    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0
    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def resize(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print(img.shape)

def combine_human(view_path, view_id, output_path, angle, action, gen, frame, x_offset, y_offset, scale_percent, img_list):

    img_path = "360degree_human/generation_pi180_{}/action{}_generation_{}/frame{}.jpg".format(angle, action, gen, frame)
    #print(img_path)
    background_path = os.path.join(view_path, view_id) + ".jpg"
    #print(background_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    img[thresh == 255] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.erode(img, kernel, iterations = 1)
   
    alpha = np.sum(result, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    res = np.dstack((result, alpha))

    background = cv2.imread(background_path)
    overlay = res.copy()

    # resize(overlay, 60)
    width = int(overlay.shape[1] * scale_percent / 100)
    height = int(overlay.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(overlay, dim, interpolation = cv2.INTER_AREA)
    #print(resized.shape)

    img = background.copy()
    add_transparent_image(img, resized, x_offset, y_offset)
    cv2.imwrite(os.path.join(output_path, view_id) + ".jpg", img)
    #print(os.path.join(output_path, view_id) + ".jpg")

def convert(heading):
    if heading > 360:
        return heading - 360
    else:
        return heading

def get_angle(rel_y, rel_x):
    if rel_x > 0 and rel_y > 0:
        angle = np.arctan2(rel_y, rel_x)
        angle = np.degrees(angle) + 180
    elif rel_x < 0 and rel_y < 0:
        angle = np.arctan2(rel_y, rel_x)
        angle = np.degrees(angle) + 180
    elif rel_x > 0 and rel_y < 0:
        angle = np.arctan2(rel_y, rel_x)
        angle = 180 - np.degrees(angle)
    else:
        angle = 360 - np.degrees(angle) 
    return angle

def compute_rel(pos_data, src_viewpoint, tar_viewpoint, current_heading):
    # convert to rel to y axis
    target_heading = np.pi/2.0 - np.arctan2(pos_data[tar_viewpoint][1] - pos_data[src_viewpoint][1], pos_data[tar_viewpoint][0] - pos_data[src_viewpoint][0]) 
    current_heading = current_heading * (2 * np.math.pi) / 360
    #print(target_heading)
    rel_heading = target_heading - current_heading
    angle = min(360, abs(target_heading) / (2 * np.math.pi) * 360)
    # normalize angle into turn into [-pi, pi]
    rel_heading = rel_heading - (2*np.pi) * np.floor((rel_heading + np.pi) / (2*np.pi))
    # rel_heading = rel_heading / np.pi * 180
    rel_heading = rel_heading / (2*np.pi) + 0.6
    return rel_heading, angle

def get_angle_offset_id(pos_data, heading_data, scan_id, tar_viewpoint, src_viewpoint):

    p1 = np.array([pos_data[src_viewpoint][0], pos_data[src_viewpoint][1], pos_data[src_viewpoint][2]])
    p2 = np.array([pos_data[tar_viewpoint][0], pos_data[tar_viewpoint][1], pos_data[tar_viewpoint][2]])
    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)
    #print(dist)
    try:
        current_heading = heading_data[scan_id][src_viewpoint][0]
    except:
        current_heading = 180
    #current_heading = 340
    #print(current_heading)
    rel, angle = compute_rel(pos_data, src_viewpoint, tar_viewpoint, current_heading) 

    if rel >= 0 and rel < 0.25:
        result_id = 1
        percent = rel * 4
        if rel >= 0.2:
            percent = 0.8
    elif rel >= 0.25 and rel < 0.5:
        result_id = 2
        percent = (rel - 0.25) * 4
        if rel >= 0.45:
            percent = 0.8
    elif rel >= 0.5 and rel < 0.75:
        result_id = 3
        percent = (rel - 0.5) * 4
        if rel >= 0.70:
            percent = 0.8
    else:
        result_id = 4
        percent = (rel - 0.75) * 4
        if rel >= 0.95:
            percent = 0.8
    rotate_angle = abs(angle)
    scale = min(90, 500 / dist)
    if scale <= 30:
        scale = 30

    human_image_shape = [656, 304, 3]
    env_img_shape = [1024, 1024, 3]
    height = human_image_shape[0] * scale / 100
    width = human_image_shape[1] * scale / 100
    # x_offset = env_img_shape[1] * percent - width / 2
    x_offset = env_img_shape[1] * percent
    y_offset = env_img_shape[0] - height - random.randrange(100, 150)

    return result_id, rotate_angle, scale, x_offset, y_offset

def get_angle_offset(pos_data, heading_data, scan_id, tar_viewpoint, src_viewpoint):
    src_heading = heading_data[scan_id][src_viewpoint][0]
    #print(src_heading)

    pic_list = [4, 1, 2, 3]
    result_id = 0
    
    rel_y = pos_data[src_viewpoint][1]- pos_data[tar_viewpoint][1]
    #print(rel_y)
    rel_x = pos_data[src_viewpoint][0]- pos_data[tar_viewpoint][0]
    #print(rel_x)
    angle = get_angle(rel_y, rel_x)
    #if angle < 0: angle = - angle
    #print(angle)

    if src_heading <= 90:
        if angle > src_heading and angle < src_heading + 90:
            result_id = pic_list[0]
            percent = (angle - src_heading) / 90
        elif angle > src_heading + 90 and angle < src_heading + 180:
            result_id = pic_list[1]
            percent = (angle - src_heading - 90) / 90
        elif angle > src_heading + 180 and angle < src_heading + 270:
            result_id = pic_list[2]
            percent = (angle - src_heading - 180) / 90
        else:
            result_id = pic_list[3]
            percent = (angle - src_heading - 270) / 90

    if 90 < src_heading <= 180:
        if angle > src_heading and angle < src_heading + 90:
            result_id = pic_list[0]
            percent = (angle - src_heading) / 90
        elif angle > src_heading + 90 and angle < src_heading + 180:
            result_id = pic_list[1]
            percent = (angle - src_heading - 90) / 90
        elif angle > src_heading + 180 and angle < 360:
            result_id = pic_list[2]
            percent = (angle - src_heading - 180) / 90
        elif angle >= 0 and angle < convert(src_heading + 270):
            result_id = pic_list[2]
            percent = 1 - (convert(src_heading + 270) - angle) / 90
        else:
            result_id = pic_list[3]
            percent = (angle - convert(src_heading + 270)) / 90

    if 180 < src_heading <= 270:
        if angle > src_heading and angle < src_heading + 90:
            result_id = pic_list[0]
            percent = (angle - src_heading) / 90
        elif angle > src_heading + 90 and angle < 360:
            result_id = pic_list[1]
            percent = (angle - src_heading - 90) / 90
        elif angle >= 0 and angle < convert(src_heading + 180):
            result_id = pic_list[1]
            percent = 1 - (convert(src_heading + 180) - angle) / 90
        elif angle > convert(src_heading + 180) and angle < convert(src_heading + 270):
            result_id = pic_list[2]
            percent = (angle - convert(src_heading + 180)) / 90
        else:
            result_id = pic_list[3]
            percent = (angle - convert(src_heading + 270)) / 90

    if 270 < src_heading <= 360:
        if angle > src_heading and angle < 360:
            result_id = pic_list[0]
            percent = (angle - src_heading) / 90
        elif angle >= 0 and angle < convert(src_heading + 90):
            result_id = pic_list[0]
            percent = 1 - (convert(src_heading + 90) - angle) / 90
        elif angle > convert(src_heading + 90) and angle < convert(src_heading + 180):
            result_id = pic_list[1]
            percent = (angle - convert(src_heading + 90)) / 90
        elif angle > convert(src_heading + 180) and angle < convert(src_heading + 270):
            result_id = pic_list[2]
            percent = (angle - convert(src_heading + 180)) / 90
        else:
            result_id = pic_list[3]
            percent = (angle - convert(src_heading + 270)) / 90

    #print(result_id, percent)  

    ## get angle and offsets
    p1 = np.array([pos_data[src_viewpoint][0], pos_data[src_viewpoint][1], pos_data[src_viewpoint][2]])
    p2 = np.array([pos_data[tar_viewpoint][0], pos_data[tar_viewpoint][1], pos_data[tar_viewpoint][2]])
    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)
    #print(dist)

    rotate_angle = angle
    scale = max(100, 70 / dist)

    human_image_shape = [656, 304, 3]
    env_img_shape = [1024, 1024, 3]
    height = human_image_shape[0] * scale / 100
    width = human_image_shape[1] * scale / 100
    x_offset = env_img_shape[1] * percent - width / 2
    y_offset = env_img_shape[0] - height - 100

    return result_id, rotate_angle, scale, x_offset, y_offset

## viewpoint selection
def main(args):

    global src_view, tar_view

    with open('human_view_info.json', 'r') as f:
        human_view_data = json.load(f)

    GRAPHS = 'connectivity/'

    with open(GRAPHS+'scans.txt') as f:
            scans = [scan.strip() for scan in f.readlines()]

    img_list = []
    record_dict = {}
    # for scan in scans:
    #     if human_view_data[scan] == []:
    #         pass
    #     else:
    #         with open('/usr0/home/yifei/Merge/con/con_info/{}_con_info.json'.format(scan), 'r') as f:
    #             connection_data = json.load(f)

    
    ## get skybox number
    with open("con/heading_info.json", 'r') as f:
        heading_data = json.load(f)
    
    if args.mode == 'run_all':
        scan_list = scans
    else:
        scan_list = []
        scan_list.append(args.scan)
    for scan_id in scan_list:
    #scan_id = "B6ByNegPMKs"
        print(scan_id)
        view_path = os.path.join(args.input_dir, "{}/matterport_skybox_images".format(scan_id))
        output_path = os.path.join(args.output_dir, "{}/matterport_skybox_images".format(scan_id))
        with open('con/con_info/{}_con_info.json'.format(scan_id), 'r') as f:
            connection_data = json.load(f)

        with open('con/pos_info/{}_pos_info.json'.format(scan_id), 'r') as f:
            pos_data = json.load(f)

        record_dict[scan_id] = {}

        for view_num in range(len(human_view_data[scan_id])):
            tar_view = human_view_data[scan_id][view_num]
            print(tar_view)
            action_select = random.randrange(0, 39)
            
            for num, val in connection_data.items():
                try:
                    if tar_view in val['visible']:
                        src_view = num
                        
                        info_list = []
                        result_id, rotate_angle, scale, x_offset, y_offset = get_angle_offset_id(pos_data, heading_data, scan_id, tar_view, src_view)
                        view_id = '{}_skybox{}_sami'.format(src_view, result_id)
                        #print(tar_viewpoint, src_viewpoint, int(x_offset), result_id)
                        gen_select = random.randrange(0, 9)
                        frame_select = random.randrange(0, 13)
                        rotate_angle = int(rotate_angle)
                        if rotate_angle == 0:
                            rotate_angle = 360
                        info_list.append(tar_view)
                        info_list.append(rotate_angle)
                        info_list.append(action_select)
                        info_list.append(int(x_offset))
                        info_list.append(int(y_offset))
                        info_list.append(int(scale))
                        info_list.append(result_id)
                        
                        # print(view_path, view_id, output_path, rotate_angle, action_select, gen_select, frame_select, x_offset, y_offset, scale)
                        combine_human(view_path, view_id, output_path, rotate_angle, action_select, gen_select, frame_select, int(x_offset), int(y_offset), int(scale), img_list)
                        img_list.append(os.path.join(output_path, view_id) + ".jpg")
                        try:
                            record_dict[scan_id][src_view] += (info_list)
                        except:
                            record_dict[scan_id][src_view] = []
                            record_dict[scan_id][src_view] +=(info_list)
                except:
                    print(scan_id)
                    break
    
    with open('images.txt', 'w') as f:
        f.write('\n'.join(img_list))
    # with open('add_info.json', 'w') as info_file:
    #     json.dump(record_dict, info_file, indent = 3)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='run_all')
    parser.add_argument('--scan', default='1LXtFkjw3qL')
    parser.add_argument('--input_dir', default='data/v1/scans')
    parser.add_argument('--output_dir', default='data_test/v1/scans')
    args = parser.parse_args()
    main(args)