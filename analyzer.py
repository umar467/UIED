import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()


def analyze(compos, frame_rgb):
    # count compos by category
    compos_by_category = {}
    for compo in compos:
        if compo.category not in compos_by_category:
            compos_by_category[compo.category] = 1
        else:
            compos_by_category[compo.category] += 1
    print('Compos by category:')
    print(compos_by_category)

    # count compos by size
    compos_by_size = {}
    for compo in compos:
        if compo.category == 'Text':
            continue
        if compo.width < 64:
            size = 'small'
        elif 64 <= compo.width < 128:
            size = 'medium'
        else:
            size = 'large'
        if size not in compos_by_size:
            compos_by_size[size] = 1
        else:
            compos_by_size[size] += 1
    print('Compos by size:')
    print(compos_by_size)

    # count compos by their image crop dominant color
    compos_by_color = {}
    for compo in compos:
        if compo.category == 'Text':
            continue
        # use bbox to crop image
        bbox = compo.bbox.put_bbox()
        crop = frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        #crop = frame_rgb[compo.row_min:compo.row_max, compo.col_min:compo.col_max]
        color = np.median(crop, axis=(0, 1))
        color = tuple(color)
        if color not in compos_by_color:
            compos_by_color[color] = 1
        else:
            compos_by_color[color] += 1
    print('Compos by color:')
    print(compos_by_color)

    # plot only the top 5 dominant colors
    compos_by_color = dict(sorted(compos_by_color.items(), key=lambda x: x[1], reverse=True)[:5])
    
    fig = plt.figure()
    sns.palplot(list(compos_by_color.keys()))
    fig.canvas.draw()
    palplot_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    palplot_img = palplot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    cv2.imshow('palplot', palplot_img)
    cv2.waitKey(100)

    