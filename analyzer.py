
import mkl
# mkl.set_num_threads(1)
import cv2
# cv2.setNumThreads(1)
import numpy as np
import matplotlib.pyplot as plt
from color_utils import convert as convert_color_blind
from color_utils import convert_fast as convert_color_blind_fast
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
        self.pseudo_frame_count = 0

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
        cimg = (x_img + y_img)# / 2

        return cimg

    def compare_contrast(self, img_o):
        # Currently 14 - > 1 reduction in seconds
        import time

        start = time.time()
        new_contrast = convert_color_blind_fast(img_o,2)
        end = time.time()
        print('new_contrast', end - start)

        cv2.imwrite(self.config.output_folder + '/new_color_'+str(self.pseudo_frame_count)+'.png', new_contrast)
        self.pseudo_frame_count += 1


        start = time.time()
        old_contrast = convert_color_blind(img_o,2)
        end = time.time()
        print('old_contrast', end - start)
        cv2.imwrite(self.config.output_folder + '/old_color_'+str(self.pseudo_frame_count)+'.png', new_contrast)

    def compare_color_speed(self, img_o):
        # Currently 14 - > 1 reduction in seconds
        import time

        start = time.time()
        new_contrast = self.convert_to_contrast_fast(img_o)
        end = time.time()
        print('new_contrast', end - start)
        new_contrast = self.get_visual_raw_contrast(new_contrast)
        cv2.imwrite(self.config.output_folder + '/new_color_'+str(self.pseudo_frame_count)+'.png', new_contrast)
        #self.pseudo_frame_count += 1
        # cv2.imshow('new_contrast', new_contrast)
        # cv2.waitKey(10)

        start = time.time()
        old_contrast = self.convert_to_contrast(img_o)
        end = time.time()
        print('old_contrast', end - start)
        old_contrast = self.get_visual_raw_contrast(old_contrast)
        cv2.imwrite(self.config.output_folder + '/old_color_'+str(self.pseudo_frame_count)+'.png', new_contrast)
        # cv2.imshow('old_contrast', old_contrast)
        # cv2.waitKey(10)



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

        cv2.imshow('oldcontrast', element_crop)
        cv2.waitKey(0)
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
        contrast_frame_from_component_contrast = np.zeros(frame_rgb.shape)

        for compo in compos:
            bbox = compo.bbox.put_bbox()
            contrast = np.array(compo.contrast_scores).mean()
            cb_contrast = np.array(compo.contrast_cb_scores).mean()
            if contrast < 0.10:
                color = [255, 0, 0]  # Blue in BGR
            if contrast >= 0.10:
                color = [0, 255, 0]  # Green in BGR
            percent_contrast_difference = abs(contrast - cb_contrast) / contrast
            #print(percent_contrast_difference)
            if abs(contrast - cb_contrast) > self.config.max_cblind_contrast_delta:
                #print(abs(contrast - cb_contrast))
                warning = {'warning_type': 'Contrast Diff. b/w CB & RGB', 'bbox': bbox,
                                  'frames_occurs_in': frame_number, 'component_id': compo.id,
                                'contrast_difference': abs(contrast - cb_contrast), 'contrast': contrast, 'cb_contrast': cb_contrast}
                JSON_Processor.warnings.append(warning)
                color = [50, 127, 205] # Brown in BGR
                self.save_cb_rgb_crop(compo, frame_rgb, cb_frame)
                color = [0,0,255]
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

    def analyze_show(self, compos, frame_rgb, frame_count, DB_Compos, config, detection_frame, JSON_Processor, current_frame_number, video_reader_object):
        just_text_check = True
        if just_text_check:
            cv2.imshow('Errors', self.check_small_text(compos, frame_rgb, JSON_Processor, frame_count))
            cv2.waitKey(100)
        else:
            frame_count = int(frame_count)
            self.config = config


            #contrast_frame = self.compare_contrast(frame_rgb)
            contrast_frame = self.convert_to_contrast_fast(frame_rgb)
            cb_frame = convert_color_blind_fast(frame_rgb, 0)
            # cb_frame = convert_color_blind(frame_rgb, 2)
            contrast_cb_frame = self.convert_to_contrast_fast(cb_frame)
            #contrast_cb_frame_show = self.get_visual_raw_contrast(contrast_cb_frame_raw)

            high_res_rgb_frame = video_reader_object.get_specific_frame(current_frame_number, downsampling=False)
            contrast_frame = self.convert_to_contrast_fast(high_res_rgb_frame)
            cb_frame = convert_color_blind_fast(high_res_rgb_frame, 0)
            cb_contrast_frame = self.convert_to_contrast_fast(cb_frame)

            cdelta = abs(contrast_frame - cb_contrast_frame) * 10
            cdelta = self.get_visual_raw_contrast(cdelta)

            contrast_frame = self.get_visual_raw_contrast(contrast_frame)

            cb_contrast_frame = self.get_visual_raw_contrast(cb_contrast_frame)
            running_id = 0
            for compo in compos:
                # score, cb_score = self.get_component_contrast_color_based(compo, frame_rgb, current_frame_number,
                #                                                           video_reader_object, compos, cb_frame,
                #                                                           contrast_frame, cb_contrast_frame,
                #                                                           high_res_rgb_frame, running_id)
                score, cb_score = self.enhanced_get_component_contrast_color_based(compo, frame_rgb, current_frame_number,
                                                                          video_reader_object, compos, cb_frame,
                                                                          contrast_frame, cb_contrast_frame,
                                                                          high_res_rgb_frame, running_id)
                #score = self.get_component_contrast(compo, contrast_frame)
                compo.contrast_scores.append(score)
                #cb_score = self.get_component_contrast(compo, contrast_cb_frame)
                compo.contrast_cb_scores.append(cb_score)
                running_id += 1
            # print the running_id right now


            contrast_frame_from_component_contrast = self.get_contrast_frame_from_component_contrast(compos, frame_rgb, JSON_Processor, frame_count, cb_frame)
            self.save_contrast_examples(compos, frame_rgb, JSON_Processor, frame_count)
            #visual_raw_contrast = self.get_visual_raw_contrast(contrast_frame)
            #visual_raw_contrast = self.show_contrast_raw(visual_raw_contrast, frame_count)
            #visual_raw_contrast = self.process_raw_visual_contrast(compos, visual_raw_contrast, JSON_Processor, frame_count, cb_frame)

            # cv2.imwrite(self.config.output_folder + '/raw_contrast_bboxed_' + str(frame_count) + '.png',
            #             visaul_raw_contrast)
            # cv2.imwrite(self.config.output_folder + '/contrast_'+str(frame_count)+'.png', contrast_frame_from_component_contrast)


            #cv2.imwrite(self.config.output_folder + '/frame_'+str(frame_count)+'.png', frame_rgb)
            #cv2.imwrite(self.config.output_folder + '/cb_frame_'+str(frame_count)+'.png', cb_frame)
            # converted_frame = np.hstack([cb_frame ,contrast_frame_from_component_contrast, visual_raw_contrast])
            # cv2.imwrite(self.config.output_folder + '/cblind_check_'+str(frame_count)+'.png', converted_frame)

            # cv2.imshow('frame', frame_rgb)
            # cv2.imshow('cb_frame', cb_frame)
            # cv2.imshow('contrast_frame', contrast_frame)
            # cv2.imshow('cbl_cont', contrast_cb_frame)
            #
            # cv2.imshow('dc', cdelta)
            # cv2.imshow('cfra_com_con', contrast_frame_from_component_contrast)
            # cv2.waitKey(0)

            # the im show command is not working in headless mode

    def _expand_bbox(self, bbox, expansion_ratio=0.2):
        """Expand a bounding box by a certain ratio."""
        x, y, xmax, ymax = bbox
        w = xmax - x
        h = ymax - y

        delta = min(w, h) * expansion_ratio

        return int(x - delta), int(y - delta), int(xmax + delta), int(ymax + delta)

    def _get_high_res_bbox(self, low_res_shape, high_res_shape, low_res_bbox):
        """Convert a low resolution bounding box to a high resolution one."""
        x, y, xmax, ymax = low_res_bbox
        xn, yn, xmaxn, ymaxn = (int(x * high_res_shape[1] / low_res_shape[1]),
        int(y * high_res_shape[0] / low_res_shape[0]),
        int(xmax * high_res_shape[1] / low_res_shape[1]),
        int(ymax * high_res_shape[0] / low_res_shape[0]))

        if xn ==0 or xn <0:
            xn = 0
        if yn ==0 or yn <0:
            yn = 0

        new =  [xn, yn, xmaxn, ymaxn]

        return new

    def _crop_frames(self, bbox, *frames):
        """Crop a series of frames using the same bounding box."""
        return [frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] for frame in frames]

    def _calculate_scores(self, contrast_crop, cb_contrast_crop):
        """Calculate the contrast score and cb contrast score."""
        white = np.count_nonzero(contrast_crop > self.config.AA_contrast_ratio)
        black = np.count_nonzero(contrast_crop < self.config.AA_contrast_ratio) + 1
        score = white / black

        white = np.count_nonzero(cb_contrast_crop > self.config.AA_contrast_ratio)
        black = np.count_nonzero(cb_contrast_crop < self.config.AA_contrast_ratio) + 1
        cb_score = white / black

        return score, cb_score

    def _save_images(self, contrast_crop, cb_crop, frame_number, running_id):
        """Save the contrast crop and cb crop images."""
        cv2.imwrite(f"{self.config.output_folder}/contrast_crop_{frame_number}_{running_id}.png", contrast_crop)
        cv2.imwrite(f"{self.config.output_folder}/cb_crop_{frame_number}_{running_id}.png", cb_crop)

    def _get_std_filtered_frame(self, video_reader, frame_number, high_res_frame):
        """Get the standard deviation filtered frame."""
        neighbour_frames = video_reader.get_neighbours_of_specific_frame(frame_number, 10, downsampling=False)
        std = np.std(np.array(neighbour_frames), axis=0)
        std = np.mean(std, axis=2)
        std = std * (std < std.mean())
        std = np.stack([std, std, std], axis=2)

        return std * high_res_frame

    def _save_std_filtered_element_crop(self, std_filtered_element_crop, frame_number, running_id):
        """Save the standard deviation filtered element crop image."""
        cv2.imwrite(f"{self.config.output_folder}/std_filtered_element_crop_{frame_number}_{running_id}.png",
                    std_filtered_element_crop)

    def _save_full_res_element_crop(self, full_res_element_crop, frame_number, running_id):
        """Save the full resolution element crop image."""
        cv2.imwrite(f"{self.config.output_folder}/full_res_element_crop_{frame_number}_{running_id}.png",
                    full_res_element_crop)

    def _detect_components_in_crop(self, full_res_element_crop, frame_number, running_id):
        """Detect components in the full resolution element crop."""
        import detect_compo.lib_ip.ip_detection as det
        import detect_compo.lib_ip.ip_preprocessing as pre

        _, board = det.component_detection_simplified_bfs(pre.convert_rgb_frame_to_binary(full_res_element_crop))
        cv2.imwrite(f"{self.config.output_folder}/board_{frame_number}_{running_id}.png", board)

    def _apply_clahe(self, img, clip_limit=3.0, tile_grid_size=(8, 8)):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))

        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    def enhanced_get_component_contrast_color_based(self, compo, rgb_frame, current_frame_number, video_reader_object, compos, cb_frame, contrast_frame, cb_contrast_frame, high_res_rgb_frame, running_id):


        # Get the bounding box of the component
        orig_bbox = compo.bbox.put_bbox()

        # Increase the bounding box boundary by 20% to get a better crop
        x, y, xmax, ymax = self._expand_bbox(orig_bbox, expansion_ratio=0.2)

        # Crop the element from the RGB frame
        element_crop = rgb_frame[y:ymax, x:xmax]

        # Get the high resolution bounding box
        high_res_bbox = self._get_high_res_bbox(rgb_frame.shape, high_res_rgb_frame.shape, [x, y, xmax, ymax])

        # Crop the element from the high resolution RGB frame, contrast frame, cb frame, and cb contrast frame
        full_res_element_crop, contrast_crop, cb_crop, cb_contrast_crop = self._crop_frames(high_res_bbox, high_res_rgb_frame, contrast_frame, cb_frame, cb_contrast_frame)

        # Calculate the contrast score and cb contrast score
        score, cb_score = self._calculate_scores(contrast_crop, cb_contrast_crop)

        # Save the contrast crop and cb crop images
        self._save_images(contrast_crop, cb_crop, current_frame_number, running_id)

        # Get the standard deviation filtered frame
        std_filtered_frame = self._get_std_filtered_frame(video_reader_object, current_frame_number, high_res_rgb_frame)

        # Crop the standard deviation filtered frame
        std_fileterd_element_crop = std_filtered_frame[high_res_bbox[1]:high_res_bbox[3], high_res_bbox[0]:high_res_bbox[2]]

        # Calculate the mean of the standard deviation filtered element crop
        std_filtered_element_crop_mean = std_fileterd_element_crop.mean()

        # Save the standard deviation filtered element crop image
        self._save_std_filtered_element_crop(std_fileterd_element_crop, current_frame_number, running_id)

        # Save the full resolution element crop image
        self._save_full_res_element_crop(full_res_element_crop, current_frame_number, running_id)

        # Detect components in the full resolution element crop
        self._detect_components_in_crop(full_res_element_crop, current_frame_number, running_id)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the full resolution element crop
        final = self._apply_clahe(full_res_element_crop)

        return score, cb_score

    def get_component_contrast_color_based(self, compo, rgb_frame, current_frame_number, video_reader_object, compos, cb_frame, contrast_frame, cb_contrast_frame, high_res_rgb_frame, running_id):


        orig_bbox = compo.bbox.put_bbox()

        bbox = orig_bbox
        # increase bbox boundary by 20% to get a better crop
        x = bbox[0]
        y = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]

        x_delta = w*0.2
        y_delta = h*0.2
        delta = min(x_delta, y_delta)

        x = x - delta
        y = y - delta
        xmax = xmax + delta
        ymax = ymax + delta

        x = int(x)
        y = int(y)
        xmax = int(xmax)
        ymax = int(ymax)

        bbox = [x, y, xmax, ymax]

        # bbox = [int(bbox[0] - bbox[2] * 0.2), int(bbox[1] - bbox[3] * 0.2), int(bbox[2] * 1.4), int(bbox[3] * 1.4)]
        #
        # bbox = [int(bbox[0] - bbox[2] * 0.2), int(bbox[1] - bbox[3] * 0.2), int(bbox[2] + bbox[2] * 0.2),
        #         int(bbox[3] + bbox[3] * 0.2)]

        element_crop = rgb_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]





        low_res_image_shape = rgb_frame.shape
        high_res_image_shape = high_res_rgb_frame.shape
        low_res_bbox = bbox
        # convert low_res_bbox to high_res_bbox
        high_res_bbox = [int(low_res_bbox[0] * high_res_image_shape[1] / low_res_image_shape[1]),
                            int(low_res_bbox[1] * high_res_image_shape[0] / low_res_image_shape[0]),
                            int(low_res_bbox[2] * high_res_image_shape[1] / low_res_image_shape[1]),
                            int(low_res_bbox[3] * high_res_image_shape[0] / low_res_image_shape[0])]


        full_res_element_crop = high_res_rgb_frame[high_res_bbox[1]:high_res_bbox[3], high_res_bbox[0]:high_res_bbox[2]]
        contrast_crop = contrast_frame[high_res_bbox[1]:high_res_bbox[3], high_res_bbox[0]:high_res_bbox[2]]
        cb_crop = cb_frame[high_res_bbox[1]:high_res_bbox[3], high_res_bbox[0]:high_res_bbox[2]]
        cb_contrast_crop = cb_contrast_frame[high_res_bbox[1]:high_res_bbox[3], high_res_bbox[0]:high_res_bbox[2]]
        # cv2.imshow('contrast_frame', contrast_frame)
        # cv2.waitKey(10)
        cv2.imwrite(self.config.output_folder + '/contrast_frame_'+str(current_frame_number)+'.png', contrast_frame)
        # cv2.imshow('contrast_crop', contrast_crop)
        # cv2.waitKey(10)
        # cv2.imshow('cb_contrast_crop', cb_contrast_crop)
        # cv2.waitKey(10)

        white = 0
        black = 0
        white = np.count_nonzero(contrast_crop[contrast_crop > self.config.AA_contrast_ratio])
        black = np.count_nonzero(contrast_crop[contrast_crop < self.config.AA_contrast_ratio])
        black += 1
        score = white / black

        white= 0
        black =0
        white = np.count_nonzero(cb_contrast_crop[cb_contrast_crop > self.config.AA_contrast_ratio])
        black = np.count_nonzero(cb_contrast_crop[cb_contrast_crop < self.config.AA_contrast_ratio])
        black += 1
        cb_score = white / black

        cv2.imwrite(self.config.output_folder + '/contrast_crop_'+str(current_frame_number)+'_'+str(running_id)+'.png', contrast_crop)
        # cv2.imshow('cb_crop', cb_crop)
        # cv2.waitKey(10)
        cv2.imwrite(self.config.output_folder + '/cb_crop_'+str(current_frame_number)+'_'+str(running_id)+'.png', cb_crop)
        # cv2.imshow('cblind_frame', cb_frame)
        # cv2.waitKey(10)
        neighbour_frames = video_reader_object.get_neighbours_of_specific_frame(current_frame_number, 10, downsampling=False)
        neighbour_frames = np.array(neighbour_frames)
        # neighbour_frames = neighbour_frames >>2
        # neighbour_frames = neighbour_frames <<2
        std = np.std(neighbour_frames, axis=0)
        std = np.mean(std, axis=2)
        std = std*(std<(std.mean()))
        std = std>0
        std = np.stack([std, std, std], axis=2)
        # min_std_rgb = ndimage.minimum_filter(std_rgb, size=20)
        std_filtered_frame = std*high_res_rgb_frame
        # cv2.imshow('std', std_filtered_frame)
        # cv2.waitKey(10)
        #cv2.imwrite(self.config.output_folder + '/std_filtered_frame_'+str(current_frame_number)+'.png', std_filtered_frame)
        std_fileterd_element_crop = std_filtered_frame[high_res_bbox[1]:high_res_bbox[3], high_res_bbox[0]:high_res_bbox[2]]
        std_filtered_element_crop_mean = std_fileterd_element_crop.mean()
        # print(std_filtered_element_crop_mean)
        std_filtered_element_crop = cv2.putText(std_fileterd_element_crop, str(std_filtered_element_crop_mean), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.imshow('std_crop', std_fileterd_element_crop)
        # cv2.waitKey(10)
        #cv2.imwrite(self.config.output_folder + '/std_filtered_element_crop_'+str(current_frame_number)+'_'+str(running_id)+'.png', std_fileterd_element_crop)

        orig_crop = rgb_frame[orig_bbox[1]:orig_bbox[3], orig_bbox[0]:orig_bbox[2]]
        # cv2.imshow('orig_crop', orig_crop)
        # cv2.waitKey(10)

        # cv2.imshow('full_res_compo_crop', full_res_element_crop)
        # cv2.waitKey(10)
        cv2.imwrite(self.config.output_folder + '/full_res_element_crop_'+str(current_frame_number)+'_'+str(running_id)+'.png', full_res_element_crop)

        import detect_compo.lib_ip.ip_detection as det
        import detect_compo.lib_ip.ip_preprocessing as pre

        _, board = det.component_detection_simplified_bfs(pre.convert_rgb_frame_to_binary(full_res_element_crop), window_name='full_res_element_crop')
        cv2.imwrite(self.config.output_folder + '/board_'+str(current_frame_number)+'_'+str(running_id)+'.png', board)
        # det.component_detection_simplified_bfs(pre.convert_rgb_frame_to_binary(std_fileterd_element_crop), window_name='std_filtered_element_crop')
        # det.component_detection_simplified_bfs(pre.convert_rgb_frame_to_binary(orig_crop), window_name='orig_crop')

        # cv2.imshow('compo_crop', element_crop)
        # cv2.waitKey(10)

        # get_compos_pallete = self.get_compos_pallete(compos, rgb_frame)
        # get_rgb_color_pallete_frame = self.get_rgb_color_pallete_frame(compos, rgb_frame)


        # convert to LAB
        lab = cv2.cvtColor(full_res_element_crop, cv2.COLOR_BGR2LAB)
        # split the LAB image to L, A, and B channels
        l, a, b = cv2.split(lab)
        # apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # merge the CLAHE enhanced L channel with the A and B channel
        limg = cv2.merge((cl, a, b))
        # convert back to BGR
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # cv2.imshow('l', l)
        # cv2.waitKey(10)
        # cv2.imshow('cl', cl)
        # cv2.waitKey(10)
        # cv2.imshow('a', a)
        # cv2.waitKey(10)
        # cv2.imshow('b', b)
        # cv2.waitKey(10)
        # cv2.imshow('limg', limg)
        # cv2.waitKey(10)
        #
        # cv2.imshow('contrast_processing', final)
        # cv2.waitKey(10)
        return score, cb_score

    def check_small_text(self, compos, frame_rgb, JSON_Processor, frame_number):
        frame_rgb = frame_rgb.copy()
        text_small = []
        for compo in compos:
            if compo.category == 'Text':
                if compo.height < self.config.min_text_height or compo.word_width < self.config.min_text_width:
                    text_small.append(compo)
                    bbox = compo.bbox.put_bbox()
                    frame_rgb = cv2.rectangle(frame_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 5)
                    cv2.imwrite(self.config.output_folder + 'text_small.png', frame_rgb)
                    print(f'Small Text Detected! Frame: {frame_number}, Component: {compo.id}')
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
        # cv2.imshow('compo_color_pallete', palplot_img)
        # cv2.waitKey(10)
        #cv2.imwrite(self.config.output_folder + 'compo_color_pallete.png', palplot_img)
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
        # cv2.imshow('frame_color_pallete', palplot_img)
        # cv2.waitKey(10)
        # cv2.imwrite(self.config.output_folder + 'frame_color_pallete_'+str(frame_count)+'.png', palplot_img)
        return palplot_img
