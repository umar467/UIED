import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

class Analyzer:
    def __init__(self):
        self.compo_type_per_frame = []
        self.compo_size_per_frame = []
        self.compo_type_area_per_frame = []

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
        cv2.imshow('Compos by category', img)
        cv2.waitKey(100)

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
        cv2.imshow('Compos by size', img)
        cv2.waitKey(100)

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
        cv2.imshow('Compos by area', img)
        cv2.waitKey(100)

    def get_centroid_of_compo(self, compo):
        bbox = compo.bbox.put_bbox()
        centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        return centroid
    def measure_distance_between_componenets(self, compo1, compo2):
        centroid1 = self.get_centroid_of_compo(compo1)
        centroid2 = self.get_centroid_of_compo(compo2)
        distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
        return distance

    def bbox_boundary_color_analysis(self, compos, frame_rgb):
        op_mask = np.zeros((frame_rgb.shape), dtype=np.uint8)
        tr_mask = np.zeros((frame_rgb.shape), dtype=np.uint8)
        real_frame_rgb = frame_rgb.copy()
        for compo in compos:
            bbox = compo.bbox.put_bbox()
            # get average color of the bbox
            bbox_color = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            compo_bbox_color = np.median(bbox_color, axis=(0,1))

            # get four bbox around each edge of the bbox with padding and buffer
            buffer = 10
            padding = 10

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

            # compare two rgb colors
            # if np.sum(np.abs(compo_bbox_color - bbox_colors)) < 50:

            # check if bbox_colors array is nan
            if np.isnan(bbox_colors).any():
                # print('boundary color is nan')
                continue
            #check if compo_bbox_color is nan
            if np.isnan(compo_bbox_color).any():
                # print('compo color is nan')
                continue

            if all(compo_bbox_color == bbox_colors):
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
    def analyze(self, compos, frame_rgb):


        self.check_nearby_compos(compos, frame_rgb)

        self.graph_area_text_image(compos)

        self.count_compo_catagory(compos)

        self.count_compo_by_size(compos)

        # quantize frame_rgb to 3 bits per channel
        frame_rgb = np.right_shift(frame_rgb, 2)
        frame_rgb = np.left_shift(frame_rgb, 2)

        self.bbox_boundary_color_analysis(compos, frame_rgb)

        self.get_compos_pallete(compos, frame_rgb)

        self.get_rgb_color_pallete_frame(compos, frame_rgb)

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
        cv2.imshow('palplot', palplot_img)
        cv2.waitKey(100)
        cv2.imshow('frame', frame_rgb)
        cv2.waitKey(100)

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
            if compo.width < 64:
                small += 1
            elif 64 <= compo.width < 128:
                medium += 1
            else:
                large += 1
        compos_by_size = np.array([small, medium, large])
        self.compo_size_per_frame.append(compos_by_size)
        self.plot_size(self.compo_size_per_frame)

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
        self.plot_category(self.compo_type_per_frame)

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
        cv2.imshow('Centroids', blank_image)
        cv2.waitKey(1000)
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
        cv2.imshow('Densest area', ds)
        cv2.imshow('Densest area on frame', frame_show)
        cv2.waitKey(1000)

    def get_rgb_color_pallete_frame(self, compos, frame_rgb):
        # set all compo crops in frame_rgb to black
        for compo in compos:
            # use bbox to crop image
            bbox = compo.bbox.put_bbox()
            frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

        # get all unique colors in frame_rgb in a list
        unique_colors = np.unique(frame_rgb.reshape(-1, frame_rgb.shape[2]), axis=0)
        # remove black color
        unique_colors = unique_colors[1:]
        # sort colors by their frequency
        unique_colors = sorted(unique_colors, key=lambda x: np.count_nonzero(np.all(frame_rgb == x, axis=2)), reverse=True)
        # get top 5 colors
        unique_colors = unique_colors[:5]

        # make a numpy array with different blocks representing each color
        palplot_img = np.zeros((100, 100*len(unique_colors), 3), dtype=np.uint8)
        for i, color in enumerate(unique_colors):
            palplot_img[:, i*100:(i+1)*100] = np.array(color, dtype=np.uint8)
        cv2.imshow('palplot2', palplot_img)
        cv2.waitKey(100)
