import mkl
# mkl.set_num_threads(1)
import cv2
# cv2.setNumThreads(1)
import numpy as np

import detect_compo.lib_ip.ip_draw as draw
import detect_compo.lib_ip.ip_preprocessing as pre
from detect_compo.lib_ip.Component import Component
import detect_compo.lib_ip.Component as Compo
import detect_compo.lib_ip.visualize_util as visualizer
from config.CONFIG import Configuration as Config
C = Config()

def merge_intersected_corner(compos, org, is_merge_contained_ele, max_gap=(0, 0), max_ele_height=25):
    '''
    :param is_merge_contained_ele: if true, merge compos nested in others
    :param max_gap: (horizontal_distance, vertical_distance) to be merge into one line/column
    :param max_ele_height: if higher than it, recognize the compo as text
    :return:
    '''
    changed = False
    new_compos = []
    Compo.compos_update(compos, org.shape)
    for i in range(len(compos)):
        merged = False
        cur_compo = compos[i]
        for j in range(len(new_compos)):
            relation = cur_compo.compo_relation(new_compos[j], max_gap)
            # print(relation)
            # draw.draw_bounding_box(org, [cur_compo, new_compos[j]], name='b-merge', show=True)
            # merge compo[i] to compo[j] if
            # 1. compo[j] contains compo[i]
            # 2. compo[j] intersects with compo[i] with certain iou
            # 3. is_merge_contained_ele and compo[j] is contained in compo[i]
            if relation == 1 or relation == -1 or relation == 2:
                # (relation == 2 and new_compos[j].height < max_ele_height and cur_compo.height < max_ele_height) or\

                new_compos[j].compo_merge(cur_compo)
                cur_compo = new_compos[j]
                # draw.draw_bounding_box(org, [new_compos[j]], name='a-merge', show=True)
                merged = True
                changed = True
                # break
        if not merged:
            new_compos.append(compos[i])

    if not changed:
        return compos
    else:
        return merge_intersected_corner(new_compos, org, is_merge_contained_ele, max_gap, max_ele_height)


def merge_text(compos, org_shape, max_word_gad=4, max_word_height=20):
    def is_text_line(compo_a, compo_b):
        (col_min_a, row_min_a, col_max_a, row_max_a) = compo_a.put_bbox()
        (col_min_b, row_min_b, col_max_b, row_max_b) = compo_b.put_bbox()

        col_min_s = max(col_min_a, col_min_b)
        col_max_s = min(col_max_a, col_max_b)
        row_min_s = max(row_min_a, row_min_b)
        row_max_s = min(row_max_a, row_max_b)

        # on the same line
        # if abs(row_min_a - row_min_b) < max_word_gad and abs(row_max_a - row_max_b) < max_word_gad:
        if row_min_s < row_max_s:
            # close distance
            if col_min_s < col_max_s or \
                    (0 < col_min_b - col_max_a < max_word_gad) or (0 < col_min_a - col_max_b < max_word_gad):
                return True
        return False

    changed = False
    new_compos = []
    row, col = org_shape[:2]
    for i in range(len(compos)):
        merged = False
        height = compos[i].height
        # ignore non-text
        # if height / row > max_word_height_ratio\
        #         or compos[i].category != 'Text':
        if height > max_word_height:
            new_compos.append(compos[i])
            continue
        for j in range(len(new_compos)):
            # if compos[j].category != 'Text':
            #     continue
            if is_text_line(compos[i], new_compos[j]):
                new_compos[j].compo_merge(compos[i])
                merged = True
                changed = True
                break
        if not merged:
            new_compos.append(compos[i])

    if not changed:
        return compos
    else:
        return merge_text(new_compos, org_shape)


def rm_top_or_bottom_corners(components, org_shape, top_bottom_height=C.THRESHOLD_TOP_BOTTOM_BAR):
    new_compos = []
    height, width = org_shape[:2]
    for compo in components:
        (column_min, row_min, column_max, row_max) = compo.put_bbox()
        # remove big ones
        # if (row_max - row_min) / height > 0.65 and (column_max - column_min) / width > 0.8:
        #     continue
        if not (row_max < height * top_bottom_height[0] or row_min > height * top_bottom_height[1]):
            new_compos.append(compo)
    return new_compos


def rm_line_v_h(binary, show=False, max_line_thickness=C.THRESHOLD_LINE_THICKNESS):
    def check_continuous_line(line, edge):
        continuous_length = 0
        line_start = -1
        for j, p in enumerate(line):
            if p > 0:
                if line_start == -1:
                    line_start = j
                continuous_length += 1
            elif continuous_length > 0:
                if continuous_length / edge > 0.6:
                    return [line_start, j]
                continuous_length = 0
                line_start = -1

        if continuous_length / edge > 0.6:
            return [line_start, len(line)]
        else:
            return None

    def extract_line_area(line, start_idx, flag='v'):
        for e, l in enumerate(line):
            if flag == 'v':
                map_line[start_idx + e, l[0]:l[1]] = binary[start_idx + e, l[0]:l[1]]

    map_line = np.zeros(binary.shape[:2], dtype=np.uint8)
    cv2.imshow('binary', binary)

    width = binary.shape[1]
    start_row = -1
    line_area = []
    for i, row in enumerate(binary):
        line_v = check_continuous_line(row, width)
        if line_v is not None:
            # new line
            if start_row == -1:
                start_row = i
                line_area = []
            line_area.append(line_v)
        else:
            # checking line
            if start_row != -1:
                if i - start_row < max_line_thickness:
                    # binary[start_row: i] = 0
                    # map_line[start_row: i] = binary[start_row: i]
                    print(line_area, start_row, i)
                    extract_line_area(line_area, start_row)
                start_row = -1

    height = binary.shape[0]
    start_col = -1
    for i in range(width):
        col = binary[:, i]
        line_h = check_continuous_line(col, height)
        if line_h is not None:
            # new line
            if start_col == -1:
                start_col = i
        else:
            # checking line
            if start_col != -1:
                if i - start_col < max_line_thickness:
                    # binary[:, start_col: i] = 0
                    map_line[:, start_col: i] = binary[:, start_col: i]
                start_col = -1

    binary -= map_line

    if show:
        cv2.imshow('no-line', binary)
        cv2.imshow('lines', map_line)
        cv2.waitKey()


def rm_line(binary_orig,
            max_line_thickness=C.THRESHOLD_LINE_THICKNESS,
            min_line_length_ratio=C.THRESHOLD_LINE_MIN_LENGTH,
            show=False, wait_key=0):
    def is_valid_line(line):
        line_length = 0
        line_gap = 0
        for j in line:
            if j > 0:
                if line_gap > 5:
                    return False
                line_length += 1
                line_gap = 0
            elif line_length > 0:
                line_gap += 1
        if line_length / width > 0.95:
            return True
        return False
    binary = binary_orig.copy()
    height, width = binary.shape[:2]
    board = np.zeros(binary.shape[:2], dtype=np.uint8)

    start_row, end_row = -1, -1
    check_line = False
    check_gap = False
    for i, row in enumerate(binary):
        # line_ratio = (sum(row) / 255) / width
        # if line_ratio > 0.9:
        if is_valid_line(row):
            # new start: if it is checking a new line, mark this row as start
            if not check_line:
                start_row = i
                check_line = True
        else:
            # end the line
            if check_line:
                # thin enough to be a line, then start checking gap
                if i - start_row < max_line_thickness:
                    end_row = i
                    check_gap = True
                else:
                    start_row, end_row = -1, -1
                check_line = False
        # check gap
        if check_gap and i - end_row > max_line_thickness:
            binary[start_row: end_row] = 0
            start_row, end_row = -1, -1
            check_line = False
            check_gap = False

    if (check_line and (height - start_row) < max_line_thickness) or check_gap:
        binary[start_row: end_row] = 0

    if show:
        cv2.imshow('no-line', binary)
        if wait_key is not None:
            cv2.waitKey(wait_key)
    return binary

def rm_noise_compos(compos):
    compos_new = []
    for compo in compos:
        if compo.category == 'Noise':
            continue
        compos_new.append(compo)
    return compos_new


def rm_noise_in_large_img(compos, org,
                      max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE):
    row, column = org.shape[:2]
    remain = np.full(len(compos), True)
    new_compos = []
    for compo in compos:
        if compo.category == 'Image':
            for i in compo.contain:
                remain[i] = False
    for i in range(len(remain)):
        if remain[i]:
            new_compos.append(compos[i])
    return new_compos


def detect_compos_in_img(compos, binary, org, max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE, show=False):
    compos_new = []
    row, column = binary.shape[:2]
    for compo in compos:
        if compo.category == 'Image':
            compo.compo_update_bbox_area()
            # org_clip = compo.compo_clipping(org)
            # bin_clip = pre.binarization(org_clip, show=show)
            bin_clip = compo.compo_clipping(binary)
            bin_clip = pre.reverse_binary(bin_clip, show=show)

            compos_rec, compos_nonrec = component_detection(bin_clip, test=False, step_h=10, step_v=10, rec_detect=True)
            for compo_rec in compos_rec:
                compo_rec.compo_relative_position(compo.bbox.col_min, compo.bbox.row_min)
                if compo_rec.bbox_area / compo.bbox_area < 0.8 and compo_rec.bbox.height > 20 and compo_rec.bbox.width > 20:
                    compos_new.append(compo_rec)
                    # draw.draw_bounding_box(org, [compo_rec], show=True)

            # compos_inner = component_detection(bin_clip, rec_detect=False)
            # for compo_inner in compos_inner:
            #     compo_inner.compo_relative_position(compo.bbox.col_min, compo.bbox.row_min)
            #     draw.draw_bounding_box(org, [compo_inner], show=True)
            #     if compo_inner.bbox_area / compo.bbox_area < 0.8:
            #         compos_new.append(compo_inner)
    compos += compos_new


def compo_filter(compos, min_area):
    compos_new = []
    for compo in compos:
        if compo.height * compo.width < min_area:
            continue
        ratio_h = compo.width / compo.height
        ratio_w = compo.height / compo.width
        if ratio_h > C.maximum_height_ratio or ratio_w > C.maximum_width_ratio or \
                (min(compo.height, compo.width) < C.minimum_component_height or max(ratio_h, ratio_w) > C.maximum_component_ratio):
            continue
        if compo.height > C.maximum_component_height or compo.width > C.maximum_component_width:
            continue
        bbox = compo.bbox.put_bbox()
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        compos_new.append(compo)
    return compos_new

def filter_components_using_static_points(components, binary, static_points):
    filtered_components = []
    static_point_image = visualizer.visualize_points(binary, static_points, rgb=False, show=False)
    for component in components:
        bbox = component.put_bbox()
        col_min, row_min, col_max, row_max = bbox
        crop = static_point_image[row_min:row_max, col_min:col_max]
        if crop.mean() > 1:
            filtered_components.append(component)
    return filtered_components


def remove_text_detections_from_binary_image(binary, text_compos):
    text_compos_visualization = visualizer.visualize_components(binary, text_compos, rgb=False, show=False, fill=True, offset=-10)
    text_compos_visualization[text_compos_visualization>0]=1
    text_compos_visualization = 1 - text_compos_visualization
    filtered_binary = binary*text_compos_visualization
    # cv2.imshow('text_filtering', filtered_binary)
    # cv2.waitKey(100)
    return filtered_binary
def plot_detection_statistics(statistics, config):
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.ion()
    p = pd.DataFrame(statistics, columns=['total_detected', 'area_filtered', 'overlap_filtered', 'sift_filtered'])
    plot = p.plot();
    #plot.title('SIFT Features across Frames')
    plot.set_xlabel("Frames x 10")
    plot.set_ylabel("Frequency")
    fig = plot.get_figure()
    video_name = 'video_' + str(config.input_video)
    video_name = video_name.replace('/', '_')
    video_name = video_name.replace('.mp4', '/')
    import os
    if not os.path.exists(video_name):
        os.mkdir(video_name)
    fig.savefig(video_name + "compo.png")
    plt.close()
def detect_components_from_binary_image(binary, static_pixels, JSON_Processor, detected_text_components = None, Text_Statistics = None, config = None, rgb_frame=None):
    current_stats = []
    binary = remove_text_detections_from_binary_image(binary, detected_text_components)
    components = component_detection(binary)
    current_stats.append(len(components))
    area_filtered_components = compo_filter(components, C.min_object_area)
    current_stats.append(len(area_filtered_components))
    overlapping_filtered_components = merge_intersected_corner(area_filtered_components, binary, True)
    overlapping_filtered_components = compo_filter(overlapping_filtered_components, C.min_object_area)
    current_stats.append(len(overlapping_filtered_components))
    if static_pixels is not None:
        static_point_fileterd_components = filter_components_using_static_points(overlapping_filtered_components, binary, static_pixels)
        overlapping_filtered_components = static_point_fileterd_components
    current_stats.append(len(overlapping_filtered_components))
    JSON_Processor.add_component_filtration_statistics_to_current_frame(current_stats)
    #Text_Statistics.append(current_stats)
    #plot_detection_statistics(Text_Statistics, config)
    for compo in overlapping_filtered_components:
        bbox = compo.bbox.put_bbox()
        compo.imcrop = rgb_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    return overlapping_filtered_components

# take the binary image as input
# calculate the connected regions -> get the bounding boundaries of them -> check if those regions are rectangles
# return all boundaries and boundaries of rectangles
def component_detection(binary, min_obj_area =C.min_object_area,
                        line_thickness=C.THRESHOLD_LINE_THICKNESS,
                        min_rec_evenness=C.THRESHOLD_REC_MIN_EVENNESS,
                        max_dent_ratio=C.THRESHOLD_REC_MAX_DENT_RATIO,
                        step_h = 5, step_v = 2,
                        rec_detect=False, show=False, test=False):
    """
    :param binary: Binary image from pre-processing
    :param min_obj_area: If not pass then ignore the small object
    :param min_obj_perimeter: If not pass then ignore the small object
    :param line_thickness: If not pass then ignore the slim object
    :param min_rec_evenness: If not pass then this object cannot be rectangular
    :param max_dent_ratio: If not pass then this object cannot be rectangular
    :return: boundary: [top, bottom, left, right]
                        -> up, bottom: list of (column_index, min/max row border)
                        -> left, right: list of (row_index, min/max column border) detect range of each row
    """
    mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
    compos_all = []
    compos_rec = []
    compos_nonrec = []
    row, column = binary.shape[0], binary.shape[1]
    for i in range(0, row, step_h):
        for j in range(i % 2, column, step_v):
            if binary[i, j] == 255 and mask[i, j] == 0:
                # get connected area
                # region = util.boundary_bfs_connected_area(binary, i, j, mask)

                mask_copy = mask.copy()
                ff = cv2.floodFill(binary, mask, (j, i), None, 0, 0, cv2.FLOODFILL_MASK_ONLY)
                if ff[0] < min_obj_area: continue
                mask_copy = mask - mask_copy
                region = np.reshape(cv2.findNonZero(mask_copy[1:-1, 1:-1]), (-1, 2))
                region = [(p[1], p[0]) for p in region]

                # filter out some compos
                component = Component(region, binary.shape)
                # calculate the boundary of the connected area
                # ignore small area
                if component.width <= C.minimum_component_width or component.height <= C.minimum_component_height:
                    continue
                if component.width > C.maximum_component_width or component.height > C.maximum_component_height:
                    continue
                # check if it is line by checking the length of edges
                # if component.compo_is_line(line_thickness):
                #     continue

                if test:
                    print('Area:%d' % (len(region)))
                    draw.draw_boundary([component], binary.shape, show=True)

                compos_all.append(component)

                if rec_detect:
                    # rectangle check
                    if component.compo_is_rectangle(min_rec_evenness, max_dent_ratio):
                        component.rect_ = True
                        compos_rec.append(component)
                    else:
                        component.rect_ = False
                        compos_nonrec.append(component)

    if show:
        print('Area:%d' % (len(region)))
        draw.draw_boundary(compos_all, binary.shape, show=True)

    draw.draw_boundary(compos_all, binary.shape, show=True)
    if rec_detect:
        return compos_rec, compos_nonrec
    else:
        return compos_all
