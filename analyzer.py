
import mkl
mkl.set_num_threads(1)
import cv2
cv2.setNumThreads(1)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import detect_compo.lib_ip.ip_preprocessing as pre
from color_utils import convert as convert_color_blind
plt.ion()

class Analyzer:
    def __init__(self, config):
        self.compo_type_per_frame = []
        self.compo_size_per_frame = []
        self.compo_type_area_per_frame = []
        self.grand_frame = np.zeros((1000, 1920, 3), np.uint8)
        self.counter_png = 0
        self.contrast_frames =[]
        self.saved_colours = False
        self.saved_contrast = False
        self.config = config

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
        fig.savefig(self.config.output_folder +"compos_by_category.png")
        plt.close()

        img = cv2.imread(self.config.output_folder +'compos_by_category.png')
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
        fig.savefig(self.config.output_folder +"compos_by_size.png")
        plt.close()
        img = cv2.imread(self.config.output_folder +'compos_by_size.png')
        # cv2.imshow('Compos by size', img)
        # cv2.waitKey(100)
        return img

    def plot_area(self, array):
        import pandas as pd
        array = np.array(array)
        p = pd.DataFrame(array, columns=['Compo_Area', 'FrameArea'])
        plot = p.plot();
        plot.set_xlabel("Frames x 10")
        plot.set_ylabel("Frequency")
        fig = plot.get_figure()
        fig.savefig(self.config.output_folder + "compos_vs_fraem_area.png")
        plt.close()

    def get_centroid_of_compo(self, compo):
        bbox = compo.bbox.put_bbox()
        centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        return centroid
    def measure_distance_between_componenets(self, compo1, compo2):
        centroid1 = self.get_centroid_of_compo(compo1)
        centroid2 = self.get_centroid_of_compo(compo2)
        distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
        return distance

    def bbox_boundary_color_analysis(self, compo, frame_rgb):
        element_crop, boundary_crop, padded_crop = self.get_element_and_boundary_crops(compo, frame_rgb)

        # reshape boundary crop to element crop shape even if it is smaller or bigger
        if boundary_crop.shape[0] < element_crop.shape[0] or boundary_crop.shape[1] < element_crop.shape[1]:
            boundary_crop = cv2.resize(boundary_crop, (element_crop.shape[1], element_crop.shape[0]))
        elif boundary_crop.shape[0] > element_crop.shape[0] or boundary_crop.shape[1] > element_crop.shape[1]:
            boundary_crop = boundary_crop[0:element_crop.shape[0], 0:element_crop.shape[1]]

        element_color_stats = self.get_color_stats(element_crop)
        boundary_color_stats = self.get_color_stats(boundary_crop)

        color_similarity = self.compare_colors(element_color_stats, boundary_color_stats)
        tc = self.convert_to_contrast(padded_crop)
        #convert to greyscale
        #tc = cv2.cvtColor(tc, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('tc', cv2.resize(tc, (128,128)))
        # cv2.imshow('element_crop', cv2.resize(element_crop, (128,128)))
        #cv2.imshow('boundary_crop', cv2.resize(boundary_crop, (128,128)))
        print('\n\n\n\n\n')
        print(color_similarity)

        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()
        white = np.count_nonzero(tc[tc > tc.mean()])
        black = np.count_nonzero(tc[tc < tc.mean()])
        black+=1
        tc = white/black
        #tc = np.mean(tc)
        # divide bu comoponent area
        tc = tc / (element_crop.shape[0] * element_crop.shape[1])
        color_similarity = tc
        print(color_similarity)

        # cframe = self.convert_to_contrast(frame_rgb)
        # cv2.imshow('cfimg', cframe)
        # cv2.waitKey(1000)

        return element_crop, color_similarity
    def convert_to_contrast(self, img):
        img = img.copy()
        # normalize img to 0-1
        img = img.astype(np.float32) / 255.0
        assert (img.max() <= 1.0 and img.min() >= 0.0)
        nimg = np.zeros((img.shape[0], img.shape[1]))
        nimgy = np.zeros((img.shape[0], img.shape[1]))
        rows = img.shape[0]
        cols = img.shape[1] - 1
        for i in range(rows):
            for j in range(0, cols, 1):
                nimg[i][j] = self.web_contrast(img[i][j], img[i][j+1])
        #nimg = nimg.astype(np.float32)/ nimg.max()
        rows = rows - 1
        for i in range(cols):
            for j in range(0, rows, 1):
                nimgy[j][i] = self.web_contrast(img[j][i], img[j][i+1])
        #nimgy = nimgy.astype(np.float32)/ nimgy.max()
        #remove every 2nd column from nimgy
        #nimgy = nimgy[:, ::2]
        #remove every 2nd row from nimg
        #nimg = nimg[::2, :]
        fimg = nimg + nimgy
        fimg = fimg /2
        # nimg = cv2.resize(nimg, (128,128))
        # fimg = cv2.resize(fimg, (128, 128))
        # nimgy = cv2.resize(nimgy, (128, 128))
        # img = cv2.resize(img, (128,128))
        # cv2.imshow('nimg', nimg)
        # cv2.imshow('fimg', fimg)
        # cv2.imshow('nimgy', nimgy)
        # cv2.imshow('img', img)
        # cv2.waitKey(10000)
        return fimg
    def web_contrast(self, c1, c2):

        def rgb(rgb1, rgb2):
            for r, g, b in (rgb1, rgb2):
                if not 0.0 <= r <= 1.0:
                    raise ValueError("r is out of valid range (0.0 - 1.0)")
                if not 0.0 <= g <= 1.0:
                    raise ValueError("g is out of valid range (0.0 - 1.0)")
                if not 0.0 <= b <= 1.0:
                    raise ValueError("b is out of valid range (0.0 - 1.0)")

            l1 = _relative_luminance(*rgb1)
            l2 = _relative_luminance(*rgb2)

            if l1 > l2:
                return (l1 + 0.05) / (l2 + 0.05)
            else:
                return (l2 + 0.05) / (l1 + 0.05)

        def _relative_luminance(r, g, b):
            r = _linearize(r)
            g = _linearize(g)
            b = _linearize(b)

            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def _linearize(v):
            if v <= 0.03928:
                return v / 12.92
            else:
                return ((v + 0.055) / 1.055) ** 2.4

        return rgb(c1, c2)
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

        padding = int(compo.width // 10)

        bbox = compo.bbox.put_bbox()
        element_crop = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        try:
            padded_crop =  frame_rgb[bbox[1]-padding:bbox[3]+padding, bbox[0]-padding:bbox[2]+padding]
        except:
            padded_crop = element_crop
        # buffer = 10


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

        return element_crop, boundary_crop, padded_crop



    def contrast_stretch(self, image):
        # Calculate the minimum and maximum pixel values
        min_val = np.min(image)
        max_val = np.max(image)

        # Perform contrast stretching
        stretched_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return stretched_image
    def compute_edge_stats(self, edge_result, frame_rgb):
        crops = []
        scores = []
        compos = []
        drawing_frame = np.zeros(frame_rgb.shape).astype(np.float64)
        for pair in edge_result:
            #pair[0] = self.contrast_stretch(pair[0])
            crops.append(pair[0])
            scores.append(pair[1])
            compos.append(pair[2])
        # sort scores
        scores_sorted = np.array(scores)
        scores_sorted = np.argsort(scores)
        #print(scores_sorted)


        # current = crops[scores_sorted[0]]
        # current = np.zeros(crops[0].shape)
        # top = cv2.resize(current, (128, 128))
        length = int(len(scores_sorted)/2)



        if not self.saved_contrast:
            output = self.config.output_folder + '/contrast_worst/'
            import os
            if not os.path.exists(output):
                os.makedirs(output)
            import random
            rand_draw = random.random()
            if rand_draw > 0:
                for i in range(length):
                    current = crops[scores_sorted[i]]
                    cv2.imwrite(output +str(i)+'.png', current)
            #self.saved_contrast = True



        if not self.saved_contrast:
            output = self.config.output_folder + '/contrast_best/'
            import os
            if not os.path.exists(output):
                os.makedirs(output)
            import random
            rand_draw = random.random()
            if rand_draw > 0:
                for i in range(length):
                    current = crops[scores_sorted[-i]]
                    cv2.imwrite(output + str(i) + '.png', current)
            #self.saved_contrast = True

        for i in range(length):
            current = crops[scores_sorted[i]]
            # sort numpy array from 0 to 255
            #cv2.imwrite(self.config.output_folder + '/contrast_bad'+str(i)+'.png', current)
            bbox = compos[scores_sorted[i]].bbox.put_bbox()

            drawing_frame = cv2.rectangle(drawing_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), -1)
            # cv2.imshow('bbottom', cv2.resize(current, (128, 128)))
            # cv2.waitKey(1000)
            current = cv2.resize(current, (128, 128))
            #top = np.hstack([top, current])
        # cv2.imshow('top', top)
        # cv2.waitKey(1000)
        #bottom = top

        # current = crops[scores_sorted[-1]]
        # current = np.zeros(crops[0].shape)
        # top = cv2.resize(current, (128, 128))
        for i in range(length):
            current = crops[scores_sorted[-i]]
            #cv2.imwrite(self.config.output_folder + '/contrast_good' + str(i) + '.png', current)
            bbox = compos[scores_sorted[-i]].bbox.put_bbox()
            drawing_frame = cv2.rectangle(drawing_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), -1)
            # cv2.imshow('topp', cv2.resize(current, (128,128)))
            # cv2.waitKey(1000)
            # current = cv2.resize(current, (128, 128))
            # top = np.hstack([top, current])


        return drawing_frame#top, bottom

    def get_component_contrast(self, compo, contrast_frame):
        bbox = compo.bbox.put_bbox()
        element_crop = contrast_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        white = np.count_nonzero(element_crop[element_crop > 3.5])
        black = np.count_nonzero(element_crop[element_crop < 3.5])
        black += 1
        score = white / black
        # tc = np.mean(tc)
        # divide bu comoponent area
        #score = score / (element_crop.shape[0] * element_crop.shape[1])
        return score
    def save_contrast_examples(self, compos, frame_rgb, JSON_Processor, frame_count):
        import os
        bad = self.config.output_folder + '/contrast_worst/'
        if not os.path.exists(bad):
            os.makedirs(bad)
        good = self.config.output_folder + '/contrast_best/'
        if not os.path.exists(good):
            os.makedirs(good)
        for compo in compos:
            bbox = compo.bbox.put_bbox()
            contrast = np.array(compo.contrast_scores).mean()
            if contrast < 0.01:
                cv2.imwrite(bad + str(compo.id) + '.png', frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                warning = {'warning_type': 'Contrast Bad', 'bbox': bbox,
                           'frames_occurs_in': frame_count, 'component_id': compo.id}
                print(warning)
                JSON_Processor.log_warning(warning)
            if contrast > 0.1:
                cv2.imwrite(good + str(compo.id) + '.png', frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]])
    def get_contrast_frame_from_component_contrast(self, compos, frame_rgb, JSON_Processor):
        contrast_frame_from_component_contrast = np.zeros(frame_rgb.shape)
        for compo in compos:
            bbox = compo.bbox.put_bbox()
            contrast = np.array(compo.contrast_scores).mean()
            cb_contrast = np.array(compo.contrast_cb_scores).mean()
            if abs(contrast - cb_contrast) > 0.017:
                print('contrast', contrast, 'cb_contrast', cb_contrast, 'delta', abs(contrast - cb_contrast))
                warning = {'warning_type': 'Contrast Diff. b/w CB & RGB', 'bbox': bbox,
                                  'frames_occurs_in': 'N/A', 'component_id': compo.id}
                JSON_Processor.warnings.append(warning)
                color = [50, 127, 205]
            else:
                if contrast >= 0.05:
                    color = [255, 0, 0]  # Blue in BGR
                if contrast >= 0.10:
                    color = [0, 255, 0]  # Green in BGR
                if contrast < 0.05:
                    color = [0, 0, 255]  # Red in BGR
            contrast_frame_from_component_contrast[bbox[1]:bbox[3], bbox[0]:bbox[2]] = color
        return contrast_frame_from_component_contrast
    def get_visual_raw_contrast(self, contrast_frame):
        visual_contrast_frame = np.zeros((contrast_frame.shape[0], contrast_frame.shape[1],3))
        visual_contrast_frame[contrast_frame >= 3.5] = [255, 0, 0]
        visual_contrast_frame[contrast_frame >= 4.5] = [0, 255, 0]
        visual_contrast_frame[contrast_frame < 3.5] = [0, 0, 0]
        return visual_contrast_frame
    def show_contrast_raw(self, contrast_frame):
        self.contrast_frames.append(contrast_frame)
        if len(self.contrast_frames)>5:
            final_contrast = np.zeros(self.contrast_frames[0].shape).astype(np.float32)
            for frame in self.contrast_frames[-5:]:
                final_contrast += frame
        else:
            final_contrast = contrast_frame
        kernel = np.ones((5, 5), np.uint8)
        final_contrast = cv2.dilate(final_contrast, kernel, iterations=1)
        cv2.imwrite(self.config.output_folder + '/contrast_raw.png', final_contrast)


    def analyze_show(self, compos, frame_rgb, frame_count, DB_Compos, config, detection_frame, JSON_Processor):
        self.config = config
        self.graph_area_text_image(compos, frame_rgb)
        contrast_frame = self.convert_to_contrast(frame_rgb)
        cb_frame = convert_color_blind(frame_rgb, 2)
        contrast_cb_frame = self.convert_to_contrast(cb_frame)
        component_contrast_score = []
        for compo in compos:
            score = self.get_component_contrast(compo, contrast_frame)
            compo.contrast_scores.append(score)
            cb_score = self.get_component_contrast(compo, contrast_cb_frame)
            compo.contrast_cb_scores.append(cb_score)
        #     component_contrast_score.append(score)
        # scores_sorted = np.argsort(component_contrast_score)
        contrast_frame_from_component_contrast = self.get_contrast_frame_from_component_contrast(compos, frame_rgb, JSON_Processor)
        self.save_contrast_examples(compos, frame_rgb, JSON_Processor, frame_count)
        visual_raw_contrast = self.get_visual_raw_contrast(contrast_frame)
        self.show_contrast_raw(visual_raw_contrast)
        cv2.imwrite(self.config.output_folder + '/contrast.png', contrast_frame_from_component_contrast)
        #print(contrast_frame.max())
        self.check_small_text(compos, frame_rgb)
        if not self.saved_colours:
            import random
            rand_draw = random.random()
            if rand_draw > 0.3:
                #converted_frame = convert_color_blind(frame_rgb, 2)
                converted_frame = np.hstack([frame_rgb, cb_frame])
                cv2.imwrite(self.config.output_folder + '/cblind_check.png', converted_frame)
                self.saved_colours = True

        # edge_result = []
        #
        # for compo in compos:
        #     element_crop, score = self.bbox_boundary_color_analysis(compo, frame_rgb)
        #     edge_result.append([element_crop, score, compo])
        #
        # text_small = self.check_small_text(compos, frame_rgb)
        # # freq = self.check_compo_frequency(compos, frame_rgb, frame_count)
        # # UI_Sets = self.show_UI_Sets(DB_Compos, frame_count)
        # self.contrast_frames.append(self.compute_edge_stats(edge_result, frame_rgb))
        #
        # final_contrast = np.zeros(self.contrast_frames[0].shape).astype(np.float32)
        # for frame in self.contrast_frames:
        #     final_contrast = final_contrast + frame
        # # cv2.imshow('cont', final_contrast)
        # # cv2.waitKey(1000)
        # # convert highest 10 percent pixel values to green colour in final_contrast
        # cv2.imwrite(self.config.output_folder + '/contrast.png', final_contrast)
        #
        # # top_edge, bottom_edge =
        # # text = np.zeros(top_edge.shape)
        # # text = cv2.putText(text, 'Top Good // Bottom Bad } Contrast', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        # #                    (255, 255, 255), 2, cv2.LINE_AA)
        # # edges = np.vstack([top_edge, text, bottom_edge])
        # # edges = pre.resize_by_height(edges, 200)
        # #self.visualize_results(frame_rgb, area_plot, count_plot, size_plot, compo_pallet_plot, frame_pallet_plot, text_small, edge_result, nearby_components, freq, detection_frame)
        #
        # if not self.saved_colours:
        #     import random
        #     rand_draw = random.random()
        #     if rand_draw > 0.3:
        #         from color_utils import convert as convert_color
        #         converted_frame = convert_color(frame_rgb, 2)
        #         converted_frame = np.hstack([frame_rgb, converted_frame])
        #         cv2.imwrite(self.config.output_folder + '/cblind_check.png', converted_frame)
        #         self.saved_colours = True
        #
        # return

    def show_UI_Sets(self, DB_Compos, video_reader):
        frame_count = video_reader.total_number_of_rgb_frames
        uis = self.compute_UI_Sets(DB_Compos, frame_count)
        #print('unique UIs: ', len(uis))
        show = video_reader.get_processed_frame()
        for ui in uis:
            show = np.hstack([show, video_reader.get_specific_frame(ui-1)])
        cv2.imwrite(self.config.output_folder+'UI_Sets'+str(self.counter_png)+'.png', show)
        # cv2.imshow('UI Sets', show)
        # cv2.waitKey(100)

    def compute_UI_Sets(self, DB_Compos, frame_count):
        all_frames = []
        for compo in DB_Compos:
            if compo.category == "Text":
                continue
            arr = compo.detected_in_frames.copy()
            frames = np.zeros((frame_count+2,))
            frames[arr] = compo.id
            all_frames.append(frames)

        # test this whole bit
        unique, uis = np.unique(all_frames, axis =1, return_index=True)
        un = unique
        un[un>0] =1
        un = un.sum(axis=0)
        un = np.argsort(un)
        if len(un) > 6:
            return un
        un = un[:6]
        uis = uis[un]
        return uis
    #
    #     all_UIs = [0]
    #     current_UI = 0
    #     for i in range(len(all_frames)):
    #         if self.UI_Delta(current_UI, i, all_frames) < 0.8:
    #             current_UI = i
    #             all_UIs.append(current_UI)
    #
    #     return all_UIs
    #
    # def UI_Delta(self, U1, U2, all_frames):
    #     if len(all_frames[U1]) < 5 or len(all_frames[U2]) < 5:
    #         return 1
    #     count = 0
    #     smaller = U1
    #     if len(all_frames[U2]) < len(all_frames[U1]):
    #         smaller = U2
    #         larger = U1
    #     else:
    #         smaller = U1
    #         larger = U2
    #     total = 0
    #     if smaller == 0:
    #         return 1
    #     assert sum(all_frames[smaller]) > 0
    #     for id in all_frames[smaller]:
    #         if id !=0:
    #             total+=1
    #             if id in all_frames[larger]:
    #                 count+=1
    #
    #     if total == 0:
    #         return 1
    #     score = int(count/total)
    #     print(score)
    #     return score
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
        #print(frequency_sorted)

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
        max = int(max)
        arr = np.zeros((max, len(freq)))
        arr[freq]=1

        # plot a 1d array as a line graph and save as a png
        plt.plot(arr)
        # plt.show()

        # save figure as a png file
        plt.savefig(self.config.output_folder +'freq.png')
        plt.close()

        img = cv2.imread(self.config.output_folder +'freq.png')
        # cv2.imshow('freq', img)
        # cv2.waitKey(1000)

        return img

    def check_small_text(self, compos, frame_rgb):
        frame_rgb = frame_rgb.copy()
        text_small = []
        for compo in compos:
            if compo.category == 'Text':
                #print(compo.height, compo.word_width)

                if compo.height < 10 or compo.word_width < 10:
                    text_small.append(compo)
                    bbox = compo.bbox.put_bbox()
                    frame_rgb = cv2.rectangle(frame_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.imwrite(self.config.output_folder + 'text_small.png', frame_rgb)
        return frame_rgb
    def contrast_measure(self, frame_rgb):
        import numpy as np
        #import cv2
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
        cv2.imwrite(self.config.output_folder + 'compo_color_pallete.png', palplot_img)
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

    def graph_area_text_image(self, compos, frame_rgb):
        text_area = 0
        image_area = 0
        for compo in compos:
            if compo.category == 'Text':
                text_area += compo.area
            else:
                image_area += compo.area
        component_area = text_area + image_area
        frame_area = frame_rgb.shape[0] * frame_rgb.shape[1]
        frame_area = frame_area - component_area
        self.compo_type_area_per_frame.append(np.array([component_area, frame_area]))
        self.plot_area(self.compo_type_area_per_frame)

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

        return frame_show, ds

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
        cv2.imwrite(self.config.output_folder + 'frame_color_pallete.png', palplot_img)
        return palplot_img
