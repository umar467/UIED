from os.path import join as pjoin
import cv2
import os


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


if __name__ == '__main__':

    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements 
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:
        1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
        2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
        3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
        4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

        mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
        web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}

        key_params = {'min-grad':10, 'ffl-block':5, 'min-ele-area':50, 'merge-contained-ele':True,
                  'max-word-inline-gap':4, 'max-line-gap':4}
    '''
    '''key_params = {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}'''
    key_params = {'min-grad':20, 'ffl-block':5, 'min-ele-area':5, 'merge-contained-ele':False,
                  'max-word-inline-gap':4, 'max-line-gap':4, 'wai_key':1}
    # set input image path
    import os
    ppp = 'data/input/frames/1/'
    old_grey = []
    old_binary=[]
    fno=-1
    lst = os.listdir(ppp)
    lst.sort()
    for filename in lst:
        print(filename)
        f = os.path.join(ppp,filename)
        input_path_img = f
        fno = fno + 1

        output_root = 'data/output/frames/1/'

        resized_height = resize_height_by_longest_edge(input_path_img)

        is_ip = True 
        is_clf = False
        is_ocr = False
        is_merge = False
        

        if is_ocr:
            import detect_text_east.ocr_east as ocr
            import detect_text_east.lib_east.eval as eval
            os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
            models = eval.load()
            ocr.east(input_path_img, output_root, models, key_params['max-word-inline-gap'],
                    resize_by_height=resized_height, show=False)

        if is_ip:
            import detect_compo.ip_region_proposal as ip
            os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
            # switch of the classification func
            classifier = None
            if is_clf:
                classifier = {}
                from cnn.CNN import CNN
                # classifier['Image'] = CNN('Image')
                classifier['Elements'] = CNN('Elements')
                # classifier['Noise'] = CNN('Noise')
            grey, binary = ip.compo_detection(input_path_img, output_root, key_params,
                            classifier=classifier, resize_by_height=resized_height, show=True, frame_no=fno,wai_key=1)
            if fno%10==0:
                if fno !=0:
                    summation = old_grey[0].astype(float)
                    for qij in range(9):
                        summation = summation + old_grey[qij]
                    import matplotlib.pyplot as plt
                    plt.imshow(summation)
                    import detect_compo.lib_ip.ip_detection as det
                    import detect_compo.lib_ip.ip_preprocessing as pre
                    import numpy as np
                    xx = np.array(summation)
                    xx /= (xx.max()/255.0)
                    xx = xx.astype(np.uint8)
                    org = cv2.cvtColor(xx, cv2.COLOR_GRAY2BGR)
                    #org =summation
                    binary = pre.binarization(org, grad_min=20, show=True, wait_key=10)
                    uicompos = det.component_detection(binary, min_obj_area=5)
                    plt.imshow(binary)
                old_grey = [grey]
                old_binary = [binary]
            else:
                old_grey.append(grey)
                old_binary.append(binary)

        if is_merge:
            import merge
            name = input_path_img.split('/')[-1][:-4]
            compo_path = pjoin(output_root, 'ip', str(name) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
            merge.incorporate(input_path_img, compo_path, ocr_path, output_root, params=key_params,
                            resize_by_height=resized_height, show=True)
