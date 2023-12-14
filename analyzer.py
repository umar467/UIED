
import mkl
mkl.set_num_threads(1)
import cv2
cv2.setNumThreads(1)
import numpy as np
import matplotlib.pyplot as plt
from color_utils import convert as convert_color_blind
import os
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

    def convert_to_contrast_fast(self, img_o):
        img = img_o.copy()
        img = img.astype(np.float32) / 255.0
        assert (img.max() <= 1.0 and img.min() >= 0.0)

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # if value in r is <= 0.03928, then r = r/12.92 if r is > 0.03928, then r = ((r+0.055)/1.055)^2.4
        r = np.where(r <= 0.03928, r / 12.92, r)
        r = np.where(r > 0.03928, ((r + 0.055) / 1.055) ** 2.4, r)
        r = r * 0.2126

        # if value in g is <= 0.03928, then g = g/12.92 if g is > 0.03928, then g = ((g+0.055)/1.055)^2.4
        g = np.where(g <= 0.03928, g / 12.92, g)
        g = np.where(g > 0.03928, ((g + 0.055) / 1.055) ** 2.4, g)
        g = g * 0.7152

        # if value in b is <= 0.03928, then b = b/12.92 if b is > 0.03928, then b = ((b+0.055)/1.055)^2.4
        b = np.where(b <= 0.03928, b / 12.92, b)
        b = np.where(b > 0.03928, ((b + 0.055) / 1.055) ** 2.4, b)
        b = b * 0.0722

        img = r + g + b

        x_img = np.zeros(img.shape)
        y_img = np.zeros(img.shape)

        # still slow part !
        [rows, cols] = img.shape
        for i in range(rows - 1):
            for j in range(cols-1):
                if img[i, j] > img[i + 1, j]:
                    x_img[i, j] = (img[i, j] + 0.05) / (img[i + 1, j] + 0.05)
                else:
                    x_img[i, j] = (img[i + 1, j] + 0.05) / (img[i, j] + 0.05)

                if img[i, j] > img[i, j + 1]:
                    y_img[i, j] = (img[i, j] + 0.05) / (img[i, j + 1] + 0.05)
                else:
                    y_img[i, j] = (img[i, j + 1] + 0.05) / (img[i, j] + 0.05)

        # parity with old code
        cimg = (x_img + y_img) / 2

        return cimg

    def compare_contrast(self, img_o):
        # Currently 14 - > 1 reduction in seconds
        import time

        start = time.time()
        new_contrast = self.convert_to_contrast_fast(img_o)
        end = time.time()
        print('new_contrast', end - start)
        new_contrast = self.get_visual_raw_contrast(new_contrast)
        cv2.imshow('new_contrast', new_contrast)
        cv2.waitKey(10)

        start = time.time()
        old_contrast = self.convert_to_contrast(img_o)
        end = time.time()
        print('old_contrast', end - start)
        old_contrast = self.get_visual_raw_contrast(old_contrast)
        cv2.imshow('old_contrast', old_contrast)
        cv2.waitKey(10)



    def convert_to_contrast(self, img):



        img = img.copy()
        img = img.astype(np.float32) / 255.0
        assert (img.max() <= 1.0 and img.min() >= 0.0)
        nimg = np.zeros((img.shape[0], img.shape[1]))
        nimgy = np.zeros((img.shape[0], img.shape[1]))
        rows = img.shape[0]
        cols = img.shape[1] - 1
        for i in range(rows):
            for j in range(0, cols, 1):
                nimg[i][j] = self.web_contrast(img[i][j], img[i][j+1])
        rows = rows - 1
        for i in range(cols):
            for j in range(0, rows, 1):
                nimgy[j][i] = self.web_contrast(img[j][i], img[j][i+1])

        fimg = nimg + nimgy
        fimg = fimg /2
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


    def get_component_contrast(self, compo, contrast_frame):
        bbox = compo.bbox.put_bbox()
        element_crop = contrast_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        white = np.count_nonzero(element_crop[element_crop > self.config.AA_contrast_ratio])
        black = np.count_nonzero(element_crop[element_crop < self.config.AA_contrast_ratio])
        black += 1
        score = white / black
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
            if contrast < self.config.compo_min_contrast_ratio:
                cv2.imwrite(bad + str(compo.id) + '.png', frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                warning = {'warning_type': 'Contrast Bad', 'bbox': bbox,
                           'frames_occurs_in': frame_count, 'component_id': compo.id, 'contrast': contrast}
                JSON_Processor.log_warning(warning)
            if contrast > self.config.compo_good_contrast_ratio:
                cv2.imwrite(good + str(compo.id) + '.png', frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]])
    def save_cb_rgb_crop(self, compo, frame_rgb, cb_frame):
        cb_rgb = self.config.output_folder + '/cb_rgb_anomaly/'
        if not os.path.exists(cb_rgb):
            os.makedirs(cb_rgb)
        bbox = compo.bbox.put_bbox()
        rgb_crop = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        cb_crop = cb_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        im = np.hstack([rgb_crop, cb_crop])
        cv2.imwrite(cb_rgb + str(compo.id) + '.png', im)

    def get_contrast_frame_from_component_contrast(self, compos, frame_rgb, JSON_Processor, frame_number, cb_frame):
        contrast_frame_from_component_contrast = frame_rgb#np.zeros(frame_rgb.shape)

        for compo in compos:
            bbox = compo.bbox.put_bbox()
            contrast = np.array(compo.contrast_scores).mean()
            cb_contrast = np.array(compo.contrast_cb_scores).mean()
            if abs(contrast - cb_contrast) > self.config.max_cblind_contrast_delta:
                warning = {'warning_type': 'Contrast Diff. b/w CB & RGB', 'bbox': bbox,
                                  'frames_occurs_in': frame_number, 'component_id': compo.id,
                                'contrast_difference': abs(contrast - cb_contrast), 'contrast': contrast, 'cb_contrast': cb_contrast}
                JSON_Processor.warnings.append(warning)
                color = [50, 127, 205] # Brown in BGR
                self.save_cb_rgb_crop(compo, frame_rgb, cb_frame)
            else:
                if contrast >= 0.05:
                    color = [255, 0, 0]  # Blue in BGR
                if contrast >= 0.10:
                    color = [0, 255, 0]  # Green in BGR
                if contrast < 0.05:
                    color = [0, 0, 255]  # Red in BGR
            contrast_frame_from_component_contrast = cv2.rectangle(contrast_frame_from_component_contrast, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            #contrast_frame_from_component_contrast[bbox[1]:bbox[3], bbox[0]:bbox[2]] = color
        return contrast_frame_from_component_contrast
    def process_raw_visual_contrast(self, compos, frame_rgb, JSON_Processor, frame_number, cb_frame):
        processed_raw_visual_contrast = np.zeros(frame_rgb.shape)

        for compo in compos:
            bbox = compo.bbox.put_bbox()
            color = [255, 255, 255]  # Red in BGR
            processed_raw_visual_contrast[bbox[1]:bbox[3], bbox[0]:bbox[2]] = frame_rgb[bbox[1]:bbox[3],
                                                                                       bbox[0]:bbox[2]]
            #processed_raw_visual_contrast = cv2.rectangle(processed_raw_visual_contrast, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
        return processed_raw_visual_contrast
    def get_visual_raw_contrast(self, contrast_frame):
        visual_contrast_frame = np.zeros((contrast_frame.shape[0], contrast_frame.shape[1],3))
        visual_contrast_frame[contrast_frame >= self.config.AA_contrast_ratio] = [255, 0, 0]
        visual_contrast_frame[contrast_frame >= self.config.AAA_contrast_ratio] = [0, 255, 0]
        visual_contrast_frame[contrast_frame < self.config.AA_contrast_ratio] = [0, 0, 0]
        return visual_contrast_frame
    def show_contrast_raw(self, contrast_frame, frame_count):
        self.contrast_frames.append(contrast_frame)
        if len(self.contrast_frames)>5:
            final_contrast = np.zeros(self.contrast_frames[0].shape).astype(np.float32)
            for frame in self.contrast_frames[-5:]:
                final_contrast += frame
        else:
            final_contrast = contrast_frame
        kernel = np.ones((5, 5), np.uint8)
        final_contrast = cv2.dilate(final_contrast, kernel, iterations=1)
        #cv2.imwrite(self.config.output_folder + '/contrast_raw_'+str(frame_count)+'.png', final_contrast)
        return final_contrast

    def analyze_show(self, compos, frame_rgb, frame_count, DB_Compos, config, detection_frame, JSON_Processor):
        frame_count = int(frame_count)
        self.config = config

        contrast_frame = self.convert_to_contrast_fast(frame_rgb)
        cb_frame = convert_color_blind(frame_rgb, 2)
        contrast_cb_frame = self.convert_to_contrast_fast(cb_frame)

        for compo in compos:
            score = self.get_component_contrast(compo, contrast_frame)
            compo.contrast_scores.append(score)
            cb_score = self.get_component_contrast(compo, contrast_cb_frame)
            compo.contrast_cb_scores.append(cb_score)

        contrast_frame_from_component_contrast = self.get_contrast_frame_from_component_contrast(compos, frame_rgb, JSON_Processor, frame_count, cb_frame)
        #self.save_contrast_examples(compos, frame_rgb, JSON_Processor, frame_count)
        visual_raw_contrast = self.get_visual_raw_contrast(contrast_frame)
        visaul_raw_contrast = self.show_contrast_raw(visual_raw_contrast, frame_count)
        visaul_raw_contrast = self.process_raw_visual_contrast(compos, visaul_raw_contrast,
                                                                                                 JSON_Processor,
                                                                                                 frame_count, cb_frame)
        # cv2.imwrite(self.config.output_folder + '/raw_contrast_bboxed_' + str(frame_count) + '.png',
        #             visaul_raw_contrast)
        # cv2.imwrite(self.config.output_folder + '/contrast_'+str(frame_count)+'.png', contrast_frame_from_component_contrast)
        #self.check_small_text(compos, frame_rgb, JSON_Processor, frame_count)

        #cv2.imwrite(self.config.output_folder + '/frame_'+str(frame_count)+'.png', frame_rgb)
        #cv2.imwrite(self.config.output_folder + '/cb_frame_'+str(frame_count)+'.png', cb_frame)
        converted_frame = np.hstack([contrast_frame_from_component_contrast, visaul_raw_contrast])
        cv2.imwrite(self.config.output_folder + '/cblind_check_'+str(frame_count)+'.png', converted_frame)


    def check_small_text(self, compos, frame_rgb, JSON_Processor, frame_number):
        frame_rgb = frame_rgb.copy()
        text_small = []
        for compo in compos:
            if compo.category == 'Text':
                if compo.height < self.config.min_text_height or compo.word_width < self.config.min_text_width:
                    text_small.append(compo)
                    bbox = compo.bbox.put_bbox()
                    frame_rgb = cv2.rectangle(frame_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.imwrite(self.config.output_folder + 'text_small.png', frame_rgb)
                    warning = {'warning_type': 'Small Text', 'bbox': bbox,
                               'frames_occurs_in': frame_number, 'component_id': compo.id,
                               'text_character_height': compo.height, 'text_character_width': compo.word_width}
                    JSON_Processor.log_warning(warning)
        return frame_rgb

    def get_compos_pallete(self, compos, frame_rgb):
        # count compos by their image crop dominant color
        compos_by_color = {}
        for compo in compos:
            bbox = compo.bbox.put_bbox()
            crop = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            color = np.median(crop, axis=(0, 1))
            color = tuple(color)
            if color not in compos_by_color:
                compos_by_color[color] = 1
            else:
                compos_by_color[color] += 1
        compos_by_color = dict(sorted(compos_by_color.items(), key=lambda x: x[1], reverse=True)[:5])
        # make a numpy array with different blocks representing each color
        palplot_img = np.zeros((100, 100 * len(compos_by_color), 3), dtype=np.uint8)
        for i, color in enumerate(compos_by_color):
            palplot_img[:, i * 100:(i + 1) * 100] = np.array(color, dtype=np.uint8)
        cv2.imwrite(self.config.output_folder + 'compo_color_pallete.png', palplot_img)
        return palplot_img

    def get_rgb_color_pallete_frame(self, compos, frame_rgb):

        # set all compo crops in frame_rgb to black
        for compo in compos:
            bbox = compo.bbox.put_bbox()
            frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

        color_array = frame_rgb.reshape(-1, 3)
        unique_colors, counts = np.unique(color_array, axis=0, return_counts=True)
        unique_colors = unique_colors[counts.argsort()]
        unique_colors = unique_colors[::-1]
        unique_colors = unique_colors[:5]

        # make a numpy array with different blocks representing each color
        palplot_img = np.zeros((100, 100*len(unique_colors), 3), dtype=np.uint8)
        for i, color in enumerate(unique_colors):
            palplot_img[:, i*100:(i+1)*100] = np.array(color, dtype=np.uint8)
        cv2.imwrite(self.config.output_folder + 'frame_color_pallete_'+str(frame_count)+'.png', palplot_img)
        return palplot_img
