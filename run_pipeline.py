from os.path import join as pjoin
import cv2
import os
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.ip_preprocessing as pre


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))

def optical_change(path):
    import cv2
    import numpy as np
    camera = cv2.VideoCapture(path)
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,4))
    kernel = np.ones((5,5),np.uint8)
    background = None
    while (True):
     ret, frame = camera.read()
     frame = cv2.resize(frame, (900,600))

     if background is None:
         background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         background = cv2.GaussianBlur(background, (21, 21), 0)
         continue
    
     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    
     diff = cv2.absdiff(background, gray_frame)
     cv2.imshow('grad',diff)
     diff = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)#cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
     cv2.imshow('diff',diff)
     #diff = cv2.dilate(diff, es, iterations = 2)
     cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
     for c in cnts:
         if cv2.contourArea(c) < 1500:
             continue
         (x, y, w, h) = cv2.boundingRect(c)
         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
     cv2.imshow("contours", frame)
     cv2.imshow("dif", diff)
     cv2.waitKey(100)
    cv2.destroyAllWindows()
    camera.release()
if __name__ == '__main__':

    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements 
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:print
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
    ppp = 'data/input/frames/11/'
    old_grey = []
    old_binary=[]
    complist = []
    skip=9
    fno=-1
    lst = os.listdir(ppp)
    lst.sort()
    wno=0
    for filename in lst:
        f = os.path.join(ppp,filename)
        input_path_img = f
        
        if skip == 1:
            skip =9
            
        else:
            skip = skip -1
            continue
        
        
        #print(filename)

        

        output_root = 'data/output/frames/1/'
        demo_out = 'data/output/frames/d/'
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
            grey, binary, uicompos, org = ip.compo_detection(input_path_img, output_root, key_params,
                            classifier=classifier, resize_by_height=resized_height, show=False, frame_no=fno,wai_key=1)
            import detect_compo.lib_ip.ip_preprocessing as pre
            if fno>0:
                g1=cv2.medianBlur(grey, 25)
                g2=cv2.medianBlur(old_grey[-1], 25)
                off = g1 - g2
                cv2.imshow('of', off)
                cv2.waitKey(100)
                #off = pre.gray_to_gradient(grey) - pre.gray_to_gradient(old_grey[-1])
                #off = pre.gray_to_gradient(off)

                import numpy as np
                hsv = np.zeros_like(org)
                hsv[...,1] = 255
                flow = cv2.calcOpticalFlowFarneback(old_grey[-1],grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            
                #cv2.imshow('mag', mag)
                # cv2.imshow('ang', ang)
                cv2.imshow('OF',rgb)
                cv2.waitKey(30)
                of = mag.sum() #off.sum()
                
                cv2.imshow('current', grey)
                cv2.imshow('old', old_grey[-1])
                cv2.imshow('frame_delta', grey - old_grey[-1])
                cv2.waitKey(10)
                #if of < 5:
                 #   continue
                #if of < 500000:
                 #   continue
                #if of > 500000:
                    #print(of)
            fno = fno + 1
            if fno%30==0:
               
                if fno !=0:
                    summation = old_grey[0].astype(float)
                    for qij in range(29):
                        summation = summation + old_grey[qij]
                    import matplotlib.pyplot as plt
                    
                    new_list = []
                    for ew in complist:
                        for re in ew:
                            new_list.append(re)
                    import detect_compo.lib_ip.ip_detection as det
                    
                    import detect_compo.lib_ip.ip_draw as draw
                    org_copy = org.copy()
                    gm = draw.draw_bounding_box(org, uicompos, show=False, name='GRAND_merged', wait_key=5)
                    cv2.imwrite(demo_out+str(wno)+'F.jpg', gm)
                    exp =draw.avgboxx(org, new_list, show=False, name='GRAND_exp', wait_key=5)
                    cv2.imshow('GRAND_exp',exp)
                    print(f'\n\n exp max, mean, min  {exp.max()} {exp.mean()} {exp.min()} \n\n')
                    
                    import numpy as np
                    xx = np.array(exp)
                    xx /= (xx.max()/255.0)
                    xx = xx.astype(np.uint8)
                    org = cv2.cvtColor(xx, cv2.COLOR_GRAY2BGR)
                    #org =summation
                    binary = pre.binarization(org, grad_min=20, show=False, wait_key=10)
                    uicompos = det.component_detection(binary, min_obj_area=5)
                    cv2.imshow('testx',binary)
                    xx=xx
                    new_list = []
                    complist = [uicompos]
                    gm = draw.draw_bounding_box(org_copy, uicompos, show=False, name='Final_MA', wait_key=5, color=(255,0,0))
                    cv2.imwrite(demo_out+str(wno)+'V.jpg', gm)
                    print(demo_out+str(wno)+'V.jpg')
                    wno=wno+1
                old_grey = [grey]
                old_binary = [binary]
                complist = [uicompos]
                fno=0
            else:
                old_grey.append(grey)
                old_binary.append(binary)
                complist.append(uicompos)
                cv2.imshow('frame_delta', grey - old_grey[fno-1])
                cv2.waitKey(10)
        
        
        if is_merge:
            import merge
            name = input_path_img.split('/')[-1][:-4]
            compo_path = pjoin(output_root, 'ip', str(name) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
            merge.incorporate(input_path_img, compo_path, ocr_path, output_root, params=key_params,
                            resize_by_height=resized_height, show=True)
    print('done')
