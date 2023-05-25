import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import detect_compo.lib_ip.ip_preprocessing as pre
plt.ion()

class Analyzer:
    def __init__(self):
        self.compo_type_per_frame = []
        self.compo_size_per_frame = []
        self.compo_type_area_per_frame = []
        self.grand_frame = np.zeros((1000, 1500, 3), np.uint8)

    def plot_category(self, array):
        import pandas as pd
        array = np.array(array)
        # make pandas dataframe
        p = pd.DataFrame(array, columns=['Text', 'Image'])
        plot = p.plot();
        # plot.title('SIFT Features across Frames')
        plot.set_xlabel("Frames x 10")
        plot.set_ylabel("Frequency")
        fig = plot.get_figure()
        fig.savefig("compos_by_category.png")
        plt.close()

        img = cv2.imread('compos_by_category.png')
        # cv2.imshow('Compos by category', img)
        # cv2.waitKey(100)
        return img

    def plot_size(self, array):
        import pandas as pd
        array = np.array(array)
        # make pandas dataframe
        p = pd.DataFrame(array, columns=['Small', 'Medium', 'Large'])
        plot = p.plot();
        # plot.title('SIFT Features across Frames')
        plot.set_xlabel("Frames x 10")
        plot.set_ylabel("Frequency")
        fig = plot.get_figure()
        fig.savefig("compos_by_size.png")
        plt.close()
        img = cv2.imread('compos_by_size.png')
        # cv2.imshow('Compos by size', img)
        # cv2.waitKey(100)
        return img

    def plot_area(self, array):
        import pandas as pd
        array = np.array(array)
        # make pandas dataframe
        p = pd.DataFrame(array, columns=['TextArea', 'ImageArea'])
        plot = p.plot();
        # plot.title('SIFT Features across Frames')
        plot.set_xlabel("Frames x 10")
        plot.set_ylabel("Frequency")
        fig = plot.get_figure()
        fig.savefig("compos_by_area.png")
        plt.close()
        img = cv2.imread('compos_by_area.png')
        # cv2.imshow('Compos by area', img)
        # cv2.waitKey(100)
        return img

    def get_centroid_of_compo(self, compo):
        bbox = compo.bbox.put_bbox()
        centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        return centroid
    def measure_distance_between_componenets(self, compo1, compo2):
        centroid1 = self.get_centroid_of_compo(compo1)
        centroid2 = self.get_centroid_of_compo(compo2)
        distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
        return distance

    def bbox_boundary_color_analysis_2(self, compos, frame_rgb):
        op_mask = np.zeros((frame_rgb.shape), dtype=np.uint8)
        tr_mask = np.zeros((frame_rgb.shape), dtype=np.uint8)
        real_frame_rgb = frame_rgb.copy()
        for compo in compos:
            bbox = compo.bbox.put_bbox()
            # get average color of the bbox
            bbox_color = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            compo_bbox_color = np.median(bbox_color, axis=(0,1))
            #self.contrast_measure(bbox_color)
            # # get four bbox around each edge of the bbox with padding and buffer
            # buffer = 10
            padding = 10
            #
            bbox_top = (bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[1])
            bbox_bottom = (bbox[0]-padding, bbox[3], bbox[2]+padding, bbox[3]+padding)
            bbox_left = (bbox[0]-padding, bbox[1], bbox[0], bbox[3])
            bbox_right = (bbox[2], bbox[1], bbox[2]+padding, bbox[3])
            bboxs = [bbox_top, bbox_bottom, bbox_left, bbox_right]
            #
            # # get the color of each bbox
            # bbox_colors = []
            # for bbox in bboxs:
            #     bbox_color = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            #     # get the average color of each bbox
            #     bbox_color = np.median(bbox_color, axis=(0,1))
            #     bbox_colors.append(bbox_color)
            #
            # # get the average color of each bbox
            # bbox_colors = np.array(bbox_colors)
            # bbox_colors = np.median(bbox_colors, axis=0)
            #
            #
            #
            # # check if bbox_colors array is nan
            # if np.isnan(bbox_colors).any():
            #     # print('boundary color is nan')
            #     continue
            # #check if compo_bbox_color is nan
            # if np.isnan(compo_bbox_color).any():
            #     # print('compo color is nan')
            #     continue
            # # compare two rgb colors
            # if np.sum(np.abs(compo_bbox_color - bbox_colors)) < 50:
            # if all(compo_bbox_color == bbox_colors):
            if np.mean(self.contrast_measure(bbox_color)) >100:
                # print('boundary color is same')
                for bbox in bboxs:
                    #cv2.rectangle(frame_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_colors, 2)
                    cv2.rectangle(tr_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), -1)
                    # cv2.imshow('frame_boundary', frame_rgb)
                    # cv2.waitKey(100)
                bbox = compo.bbox.put_bbox()
                cv2.rectangle(tr_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), -1)
            else:
                # increase bbox by padding
                bbox = compo.bbox.put_bbox()
                bbox = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
                # draw the bbox on the frame
                #cv2.rectangle(frame_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), compo_bbox_color, 2)
                cv2.rectangle(op_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), -1)
                # cv2.imshow('frame_boundary', frame_rgb)
                # cv2.waitKey(100)
        masked_op_frame = cv2.bitwise_and(real_frame_rgb, op_mask)
        cv2.imshow('masked_op_frame', masked_op_frame)
        cv2.waitKey(100)
        masked_tr_frame = cv2.bitwise_and(real_frame_rgb, tr_mask)
        cv2.imshow('masked_tr_frame', masked_tr_frame)
        cv2.waitKey(100)
    def bbox_boundary_color_analysis_3(self, compos, frame_rgb):
        op_mask = np.zeros((frame_rgb.shape), dtype=np.uint8)
        tr_mask = np.zeros((frame_rgb.shape), dtype=np.uint8)
        real_frame_rgb = frame_rgb.copy()
        for compo in compos:
            bbox = compo.bbox.put_bbox()
            # get average color of the bbox
            bbox_color = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            compo_bbox_color = np.median(bbox_color, axis=(0,1))
            #self.contrast_measure(bbox_color)
            # # get four bbox around each edge of the bbox with padding and buffer
            # buffer = 10
            padding = 10
            #
            bbox_top = (bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[1])
            bbox_bottom = (bbox[0]-padding, bbox[3], bbox[2]+padding, bbox[3]+padding)
            bbox_left = (bbox[0]-padding, bbox[1], bbox[0], bbox[3])
            bbox_right = (bbox[2], bbox[1], bbox[2]+padding, bbox[3])
            bboxs = [bbox_top, bbox_bottom, bbox_left, bbox_right]

            # get the color of each bbox
            bbox_colors = []
            for bbox in bboxs:
                bbox_color = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                # get the average color of each bbox
                bbox_color = np.median(bbox_color, axis=(0,1))
                bbox_colors.append(bbox_color)

            # get the average color of each bbox
            bbox_colors = np.array(bbox_colors)
            bbox_colors = np.median(bbox_colors, axis=0)



            # check if bbox_colors array is nan
            if np.isnan(bbox_colors).any():
                # print('boundary color is nan')
                continue
            #check if compo_bbox_color is nan
            if np.isnan(compo_bbox_color).any():
                # print('compo color is nan')
                continue
            # # compare two rgb colors
            # if np.sum(np.abs(compo_bbox_color - bbox_colors)) < 50:
            if all(compo_bbox_color == bbox_colors):
            #if np.mean(self.contrast_measure(bbox_color)) >100:
                # print('boundary color is same')
                for bbox in bboxs:
                    #cv2.rectangle(frame_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_colors, 2)
                    cv2.rectangle(tr_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), -1)
                    # cv2.imshow('frame_boundary', frame_rgb)
                    # cv2.waitKey(100)
                bbox = compo.bbox.put_bbox()
                cv2.rectangle(tr_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), -1)
            else:
                # increase bbox by padding
                bbox = compo.bbox.put_bbox()
                bbox = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
                # draw the bbox on the frame
                #cv2.rectangle(frame_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), compo_bbox_color, 2)
                cv2.rectangle(op_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), -1)
                # cv2.imshow('frame_boundary', frame_rgb)
                # cv2.waitKey(100)
        masked_op_frame = cv2.bitwise_and(real_frame_rgb, op_mask)
        cv2.imshow('masked_op_frame', masked_op_frame)
        cv2.waitKey(100)
        masked_tr_frame = cv2.bitwise_and(real_frame_rgb, tr_mask)
        cv2.imshow('masked_tr_frame', masked_tr_frame)
        cv2.waitKey(100)

    def bbox_boundary_color_analysis(self, compo, frame_rgb):
        element_crop, boundary_crop = self.get_element_and_boundary_crops(compo, frame_rgb)

        # reshape boundary crop to element crop shape even if it is smaller or bigger
        if boundary_crop.shape[0] < element_crop.shape[0] or boundary_crop.shape[1] < element_crop.shape[1]:
            boundary_crop = cv2.resize(boundary_crop, (element_crop.shape[1], element_crop.shape[0]))
        elif boundary_crop.shape[0] > element_crop.shape[0] or boundary_crop.shape[1] > element_crop.shape[1]:
            boundary_crop = boundary_crop[0:element_crop.shape[0], 0:element_crop.shape[1]]

        element_color_stats = self.get_color_stats(element_crop)
        boundary_color_stats = self.get_color_stats(boundary_crop)

        color_similarity = self.compare_colors(element_color_stats, boundary_color_stats)

        # if color_similarity > 1500:
            # cv2.imshow('element_crop', element_crop)
            # cv2.imshow('boundary_crop', boundary_crop)
            # print('\n\n\n\n\n')
            # print(color_similarity)
            # cv2.waitKey(10000)
        return element_crop, color_similarity

    def compare_colors(self, col1, col2):
        # compare average color
        avg_color_diff = np.sum(np.abs(col1[0] - col2[0]))
        # compare median color
        median_color_diff = np.sum(np.abs(col1[1] - col2[1]))
        # compare std color
        std_color_diff = np.sum(np.abs(col1[2] - col2[2]))
        # compare common colors
        common_color_diff = 0
        # print(col1[3].shape)
        # print(col2[3].shape)
        for i in range(len(col1[3])):
            common_color_diff += np.sum(np.abs(col1[3][i] - col2[3][i]))
        color_similarity = avg_color_diff + median_color_diff + std_color_diff + common_color_diff
        return color_similarity
    def get_color_stats(self, crop):
        averge_color = np.mean(crop, axis=(0, 1))
        median_color = np.median(crop, axis=(0, 1))
        std_color = np.std(crop, axis=(0, 1))
        color_array = crop.reshape(-1, 3)
        unique_colors, counts = np.unique(color_array,axis=0, return_counts = True)
        unique_colors = unique_colors[counts.argsort()]
        unique_colors = unique_colors[::-1]
        common_colors = unique_colors[0:3]
        if common_colors.shape[0] != 3:
            blanks = [[0,0,0]]*(3-common_colors.shape[0])
            common_colors = np.vstack([common_colors, blanks])
        color_stats = [averge_color, median_color, std_color, common_colors]
        return color_stats
    def get_element_and_boundary_crops(self, compo, frame_rgb):
        op_mask = np.zeros((frame_rgb.shape), dtype=np.uint8)
        tr_mask = np.zeros((frame_rgb.shape), dtype=np.uint8)
        real_frame_rgb = frame_rgb.copy()
        frame_rgb = frame_rgb.copy()

        bbox = compo.bbox.put_bbox()
        element_crop = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # buffer = 10
        padding = int(compo.width // 10)

        bbox_top = (bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[1])
        bbox_bottom = (bbox[0]-padding, bbox[3], bbox[2]+padding, bbox[3]+padding)
        bbox_left = (bbox[0]-padding, bbox[1], bbox[0], bbox[3])
        bbox_right = (bbox[2], bbox[1], bbox[2]+padding, bbox[3])

        bbox_top_crop = frame_rgb[bbox_top[1]:bbox_top[3], bbox_top[0]:bbox_top[2]]
        bbox_bottom_crop = frame_rgb[bbox_bottom[1]:bbox_bottom[3], bbox_bottom[0]:bbox_bottom[2]]
        bbox_left_crop = frame_rgb[bbox_left[1]:bbox_left[3], bbox_left[0]:bbox_left[2]]
        bbox_right_crop = frame_rgb[bbox_right[1]:bbox_right[3], bbox_right[0]:bbox_right[2]]

        try:
            vertical_crops = np.vstack([bbox_top_crop, bbox_bottom_crop])
            horizontal_crops = np.vstack([bbox_left_crop, bbox_right_crop])
            horizontal_crops = horizontal_crops.transpose(1, 0, 2)
            # print(vertical_crops.shape)
            # print(horizontal_crops.shape)
            if vertical_crops.shape[0] > horizontal_crops.shape[0]:
                horizontal_crops = np.vstack([horizontal_crops, np.zeros((vertical_crops.shape[0]-horizontal_crops.shape[0], horizontal_crops.shape[1], horizontal_crops.shape[2]), dtype=np.uint8)])
                # print('NEW DIMS !!!')
                # print(vertical_crops.shape)
                # print(horizontal_crops.shape)
            boundary_crop = np.hstack([vertical_crops, horizontal_crops])
        except:
            boundary_crop = element_crop

        # cv2.imshow('boundary_crop', boundary_crop)
        # cv2.imshow('element_crop', element_crop)
        # cv2.imshow('frame_rgb', frame_rgb)
        # cv2.waitKey(1000)

        return element_crop, boundary_crop

    def visualize_results(self, area_plot, count_plot, size_plot, compo_pallet_plot, frame_pallet_plot, frame_rgb, edge_result, nearby_components, freq):
        print(area_plot.shape)
        print(count_plot.shape)
        print(size_plot.shape)
        print(compo_pallet_plot.shape)
        print(frame_pallet_plot.shape)

        shapes = [(200, 200), (50, 100)]

        area_plot = pre.resize_by_height(area_plot, 200)
        count_plot = pre.resize_by_height(count_plot, 200)
        size_plot = pre.resize_by_height(size_plot, 200)

        compo_pallet_plot = pre.resize_by_height(compo_pallet_plot, 50)
        frame_pallet_plot = pre.resize_by_height(frame_pallet_plot, 50)

        info_plots = np.vstack([area_plot, count_plot, size_plot])

        x = 10
        y = 700
        self.grand_frame[x:x+info_plots.shape[0], y:y+info_plots.shape[1]] = info_plots

        color_plots = np.vstack([compo_pallet_plot, np.zeros(compo_pallet_plot.shape), frame_pallet_plot])
        x = 700
        y = 700
        self.grand_frame[x:x+color_plots.shape[0], y:y+color_plots.shape[1]] = color_plots

        frame_rgb = pre.resize_by_height(frame_rgb, 400)
        x = 10
        y = 10
        self.grand_frame[x:x + frame_rgb.shape[0], y:y + frame_rgb.shape[1]] = frame_rgb

        nearby_components = pre.resize_by_height(nearby_components, 400)
        x = 500
        y = 10
        self.grand_frame[x:x + nearby_components.shape[0], y:y + nearby_components.shape[1]] = nearby_components


        top_edge, bottom_edge = self.compute_edge_stats(edge_result)
        edges = np.vstack([top_edge, np.zeros(top_edge.shape), bottom_edge])
        edges = pre.resize_by_height(edges, 200)
        x = 500
        y = 300
        self.grand_frame[x:x + edges.shape[0], y:y + edges.shape[1]] = edges

        freq = np.vstack([freq[0], np.zeros(freq[0].shape), freq[1]])
        freq = pre.resize_by_height(freq, 400)
        x = 400
        y = 400
        self.grand_frame[x:x + freq.shape[0], y:y + freq.shape[1]] = freq

        cv2.imshow('grand_frame', self.grand_frame)
        cv2.waitKey(1000)

    def compute_edge_stats(self, edge_result):
        crops = []
        scores = []
        for pair in edge_result:
            crops.append(pair[0])
            scores.append(pair[1])

        # sort scores
        scores_sorted = np.array(scores)
        scores_sorted = np.argsort(scores)
        print(scores_sorted)


        # current = crops[scores_sorted[0]]
        current = np.zeros(crops[0].shape)
        top = cv2.resize(current, (128, 128))
        for i in range(5):
            current = crops[scores_sorted[i]]
            current = cv2.resize(current, (128, 128))
            top = np.hstack([top, current])
        # cv2.imshow('top', top)
        # cv2.waitKey(1000)
        bottom = top

        # current = crops[scores_sorted[-1]]
        current = np.zeros(crops[0].shape)
        top = cv2.resize(current, (128, 128))
        for i in range(5):
            current = crops[scores_sorted[-i]]
            current = cv2.resize(current, (128, 128))
            top = np.hstack([top, current])
        # cv2.imshow('bottom', top)
        # cv2.waitKey(1000)

        return top, bottom

    def analyze(self, compos, frame_rgb, frame_count):
        frame = frame_rgb.copy()
        nearby_components = self.check_nearby_compos(compos, frame_rgb)
        frame_rgb = frame.copy()
        area_plot = self.graph_area_text_image(compos)
        frame_rgb = frame.copy()
        count_plot = self.count_compo_catagory(compos)
        frame_rgb = frame.copy()
        size_plot = self.count_compo_by_size(compos)
        frame_rgb = frame.copy()
        compo_pallet_plot = self.get_compos_pallete(compos, frame_rgb)
        frame_rgb = frame.copy()
        frame_pallet_plot = self.get_rgb_color_pallete_frame(compos, frame_rgb)
        frame_rgb = frame.copy()

        # quantize frame_rgb to 3 bits per channel
        frame_rgb = np.right_shift(frame_rgb, 5)
        frame_rgb = np.left_shift(frame_rgb, 5)

        edge_result = []
        for compo in compos:
            element_crop, score = self.bbox_boundary_color_analysis(compo, frame_rgb)
            edge_result.append([element_crop, score])

        text_small = self.check_small_text(compos, frame_rgb)
        freq = self.check_compo_frequency(compos, frame_rgb, frame_count)
        self.visualize_results(area_plot, count_plot, size_plot, compo_pallet_plot, frame_pallet_plot, text_small, edge_result, nearby_components, freq)

    def check_compo_frequency(self, compos, frame_rgb, frame_count):
        frame_rgb = frame_rgb.copy()
        frequency = []
        crops = []
        for compo in compos:
            fc = np.count_nonzero(compo.detected_in_frames)
            frequency.append(fc)
            cc = compo.bbox.put_bbox()
            cc = frame_rgb[cc[1]:cc[3], cc[0]:cc[2]]
            crops.append(cc)

        # sort scores
        frequency_sorted = np.array(frequency)
        frequency_sorted = np.argsort(frequency)
        print(frequency_sorted)

        size = (128,128)

        current = np.zeros(crops[0].shape)
        top = cv2.resize(current, size)
        freq_graph = self.make_frequency_graph(compos[frequency_sorted[0]], frame_count)
        freq_graph = cv2.resize(freq_graph, size)
        top = np.vstack([top, freq_graph])
        for i in range(5):
            current = crops[frequency_sorted[i]]
            current = cv2.resize(current, size)
            freq_graph = self.make_frequency_graph(compos[frequency_sorted[i]], frame_count)
            freq_graph = cv2.resize(freq_graph, size)
            current = np.vstack([current, freq_graph])
            top = np.hstack([top, current])
        # cv2.imshow('top', top)
        # cv2.waitKey(1000)
        bottom = top

        current = np.zeros(crops[0].shape)
        top = cv2.resize(current, size)
        freq_graph = self.make_frequency_graph(compos[frequency_sorted[0]], frame_count)
        freq_graph = cv2.resize(freq_graph, size)
        top = np.vstack([top, freq_graph])
        for i in range(5):
            current = crops[frequency_sorted[-i]]
            current = cv2.resize(current, size)
            freq_graph = self.make_frequency_graph(compos[frequency_sorted[-i]], frame_count)
            freq_graph = cv2.resize(freq_graph, size)
            current = np.vstack([current, freq_graph])
            top = np.hstack([top, current])
        # cv2.imshow('top', top)
        # cv2.waitKey(1000)



        return [top, bottom]
    def make_frequency_graph(self, compo, frame_count):
        freq = compo.detected_in_frames
        max = frame_count + 1
        min = 0
        arr = np.zeros((max, len(freq)))
        arr[freq]=1

        # plot a 1d array as a line graph and save as a png
        plt.plot(arr)
        # plt.show()

        # save figure as a png file
        plt.savefig('freq.png')
        plt.close()

        img = cv2.imread('freq.png')
        # cv2.imshow('freq', img)
        # cv2.waitKey(1000)

        return img

    def check_small_text(self, compos, frame_rgb):
        frame_rgb = frame_rgb.copy()
        text_small = []
        for compo in compos:
            if compo.category == 'Text':
                print(compo.height, compo.word_width)
                if compo.height < 10 or compo.word_width < 10:
                    text_small.append(compo)
                    bbox = compo.bbox.put_bbox()
                    frame_rgb = cv2.rectangle(frame_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        return frame_rgb
    def contrast_measure(self, frame_rgb):
        import numpy as np
        import cv2
        size_n = 5  # NxN neighborhood around each pixel
        # Read input image
        img = frame_rgb.copy()  # cv2.imread('chelsea.png')
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Get the L channel
        L = lab[:, :, 0]
        # Use "dilate" morphological operation (dilate is equivalent to finding maximum pixel in NxN neighborhood)
        img_max = cv2.morphologyEx(L, cv2.MORPH_DILATE, np.ones((size_n, size_n)))
        # Use "erode" morphological operation (dilate is equivalent to finding maximum pixel in NxN neighborhood)
        img_min = cv2.morphologyEx(L, cv2.MORPH_ERODE, np.ones((size_n, size_n)))
        # Convert to type float (required before using division operation)
        img_max = img_max.astype(float)
        img_min = img_min.astype(float)
        # Compute contrast map (range of img_contrast is [0, 1])
        img_contrast = (img_max - img_min) / (img_max + img_min)
        # Convert contrast map to type uint8 with rounding - the conversion loosed accuracy, so I can't recommend it.
        # Note: img_contrast_uint8 is scaled by 255 (scaled by 255 relative to the original formula).
        img_contrast_uint8 = np.round(img_contrast * 255).astype(np.uint8)
        # mean = np.mean(img_contrast_uint8)
        #write mean as string to the image

        # resize image to 128x128
        # img_contrast_uint8 = cv2.resize(img_contrast_uint8, (128, 128))
        # tim = np.zeros(img_contrast_uint8.shape)
        # cv2.putText(tim, str(mean), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # img_contrast_uint8 = np.hstack([img_contrast_uint8, tim])
        # # normalize image between 0 and 255
        # img_contrast_uint8 = cv2.normalize(img_contrast_uint8, None, 0, 255, cv2.NORM_MINMAX)
        # Show img_contrast as output
        cv2.imshow('img_contrast', img_contrast_uint8)
        cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        return img_contrast_uint8

    def get_compos_pallete(self, compos, frame_rgb):
        # count compos by their image crop dominant color
        compos_by_color = {}
        for compo in compos:
            # use bbox to crop image
            bbox = compo.bbox.put_bbox()
            crop = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # crop = frame_rgb[compo.row_min:compo.row_max, compo.col_min:compo.col_max]
            color = np.median(crop, axis=(0, 1))
            color = tuple(color)
            if color not in compos_by_color:
                compos_by_color[color] = 1
            else:
                compos_by_color[color] += 1
        # print('Compos by color:')
        # print(compos_by_color)
        # plot only the top 5 dominant colors
        compos_by_color = dict(sorted(compos_by_color.items(), key=lambda x: x[1], reverse=True)[:5])
        # make a numpy array with different blocks representing each color
        palplot_img = np.zeros((100, 100 * len(compos_by_color), 3), dtype=np.uint8)
        for i, color in enumerate(compos_by_color):
            palplot_img[:, i * 100:(i + 1) * 100] = np.array(color, dtype=np.uint8)
        # cv2.imshow('palplot', palplot_img)
        # cv2.waitKey(100)
        # cv2.imshow('frame', frame_rgb)
        # cv2.waitKey(100)
        return palplot_img

    def count_compo_by_size(self, compos):
        # count compos by size
        compos_by_size = {}
        small = 0
        medium = 0
        large = 0
        # count compos by size
        for compo in compos:
            if compo.category == 'Text':
                continue
            if compo.width < 32:
                small += 1
            elif 32 <= compo.width < 64:
                medium += 1
            else:
                large += 1
        compos_by_size = np.array([small, medium, large])
        self.compo_size_per_frame.append(compos_by_size)
        size_plot = self.plot_size(self.compo_size_per_frame)
        return size_plot

    def count_compo_catagory(self, compos):
        # count compos by category
        compos_by_category = {}
        text = 0
        image = 0
        # count compos by category
        for compo in compos:
            if compo.category == 'Text':
                text += 1
            else:
                image += 1
        compos_by_category = np.array([text, image])
        # compos_by_category = np.append(compos_by_category, 0)
        # compos_by_category = compos_by_category[:2]
        self.compo_type_per_frame.append(compos_by_category)
        count_plot = self.plot_category(self.compo_type_per_frame)
        return count_plot

    def graph_area_text_image(self, compos):
        # measure cumulative area of text and image compos
        text_area = 0
        image_area = 0
        for compo in compos:
            if compo.category == 'Text':
                text_area += compo.area
            else:
                image_area += compo.area
        self.compo_type_area_per_frame.append(np.array([text_area, image_area]))
        area_plot = self.plot_area(self.compo_type_area_per_frame)
        return area_plot

    def check_nearby_compos(self, compos, frame_rgb):
        # plot all centroids of all components on a blank image
        blank_image = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1], 3), np.uint8)
        for compo in compos:
            centroid = self.get_centroid_of_compo(compo)
            # if the 50x50 box around the centroid is not black, then skip
            skip_block_size = 50 // 2
            if np.sum(blank_image[centroid[1] - skip_block_size:centroid[1] + skip_block_size,
                      centroid[0] - skip_block_size:centroid[0] + skip_block_size]) > 0:
                continue
            # make the pixel at the centroid white
            blank_image[centroid[1], centroid[0]] = 1
        # cv2.imshow('Centroids', blank_image)
        # cv2.waitKey(1000)
        # run a convoltion of kernel size 200x200 to find the most dense area of centroids
        location_size = 80
        kernel = np.ones((location_size, location_size), np.uint8)
        # cv2 to convolve kernel with blank_image to get resultant image
        ds = cv2.filter2D(blank_image, -1, kernel)
        # make small points in ds image white circle
        frame_show = frame_rgb.copy()
        ds[ds == 1] = 0
        # get index of all the points in ds that are not 0
        nonzero = np.nonzero(ds)
        # make all nonzero points in frame_show red
        frame_show[nonzero[0], nonzero[1], 0] = 0
        frame_show[nonzero[0], nonzero[1], 1] = 0
        frame_show[nonzero[0], nonzero[1], 2] = 255
        ds[ds > 0] = 255

        # cv2.imshow('Densest area', ds)
        # cv2.imshow('Densest area on frame', frame_show)
        # cv2.waitKey(1000)

        return frame_show

    def get_rgb_color_pallete_frame(self, compos, frame_rgb):
        # set all compo crops in frame_rgb to black
        for compo in compos:
            # use bbox to crop image
            bbox = compo.bbox.put_bbox()
            frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

        # get all unique colors in frame_rgb in a list
        # unique_colors = np.unique(frame_rgb.reshape(-1, frame_rgb.shape[2]), axis=0)
        # # remove black color
        # unique_colors = unique_colors[1:]
        # # sort colors by their frequency
        # unique_colors = sorted(unique_colors, key=lambda x: np.count_nonzero(np.all(frame_rgb == x, axis=2)), reverse=True)
        # # get top 5 colors
        # unique_colors = unique_colors[:5]

        color_array = frame_rgb.reshape(-1, 3)
        unique_colors, counts = np.unique(color_array, axis=0, return_counts=True)
        unique_colors = unique_colors[counts.argsort()]
        unique_colors = unique_colors[::-1]
        unique_colors = unique_colors[:5]

        # make a numpy array with different blocks representing each color
        palplot_img = np.zeros((100, 100*len(unique_colors), 3), dtype=np.uint8)
        for i, color in enumerate(unique_colors):
            palplot_img[:, i*100:(i+1)*100] = np.array(color, dtype=np.uint8)
        # cv2.imshow('palplot2', palplot_img)
        # cv2.waitKey(100)
        return palplot_img
