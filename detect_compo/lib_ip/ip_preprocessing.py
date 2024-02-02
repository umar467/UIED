import mkl
# mkl.set_num_threads(1)
import cv2
# cv2.setNumThreads(1)
import numpy as np
import detect_compo.lib_ip.visualize_util as visualizer
from config.CONFIG import Configuration
config = Configuration()

def resize_by_height(org, resize_height):
    w_h_ratio = org.shape[1] / org.shape[0]
    resize_w = resize_height * w_h_ratio
    re = cv2.resize(org, (int(resize_w), int(resize_height)))
    return re

def read_img(path, resize_height=None, kernel_size=None):

    try:
        img = cv2.imread(path)
        if kernel_size is not None:
            img = cv2.medianBlur(img, kernel_size)
        if img is None:
            print("*** Image does not exist ***")
            return None, None
        if resize_height is not None:
            img = resize_by_height(img, resize_height)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    except Exception as e:
        print(e)
        print("*** Img Reading Failed ***\n")
        return None, None

def conver_frames_to_grey(frames):
    grey_frames = []
    for frame in frames:
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey_frames.append(grey_frame)
    grey_frames = np.array(grey_frames)
    return grey_frames

def gray_to_gradient(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_f = np.copy(img)
    img_f = img_f.astype("float")

    kernel_h = np.array([[0,0,0], [0,-1.,1.], [0,0,0]])
    kernel_v = np.array([[0,0,0], [0,-1.,0], [0,1.,0]])
    dst1 = abs(cv2.filter2D(img_f, -1, kernel_h))
    dst2 = abs(cv2.filter2D(img_f, -1, kernel_v))
    gradient = (dst1 + dst2).astype('uint8')
    return gradient


def grad_to_binary(grad, min):
    rec, bin = cv2.threshold(grad, min, 255, cv2.THRESH_BINARY)
    return bin


def reverse_binary(bin, show=False):
    """
    Reverse the input binary image
    """
    r, bin = cv2.threshold(bin, 1, 255, cv2.THRESH_BINARY_INV)
    if show:
        cv2.imshow('binary_rev', bin)
        cv2.waitKey()
    return bin

def conver_frames_to_gradient(grey_frames):
    gradient_frames=[]
    for frame in  grey_frames:
        grad_frame = gray_to_gradient(frame)
        gradient_frames.append(grad_frame)
    gradient_frames = np.array(gradient_frames)
    return gradient_frames

def extract_common_gradients(frames):
    old_grad = frames[0]
    for frame in frames:
        #visualizer.show_frame(frame, use_cv=True, name='grad_frame')
        current_grad = old_grad & frame
    #visualizer.show_frame(current_grad, use_cv=True, name='common_grad_frame')
    return current_grad

def convert_frame_to_binary(frame):
    binary = grad_to_binary(frame, config.minimum_gradient_difference)
    morphed_binary = cv2.dilate(binary, None, iterations=config.binary_dilation_iterations)
    return morphed_binary

def binarization(org, grad_min, morphology_size, show=False, write_path=None, wait_key=0):
    grey = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    grad = gray_to_gradient(grey)        # get RoI with high gradient
    binary = grad_to_binary(grad, grad_min)   # enhance the RoI

    morph = binary
    #morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morphology_size)  # remove noises
    #morph = cv2.dilate(morph, None, iterations=7)

    if write_path is not None:
        cv2.imwrite(write_path, morph)
    if show:
        full = np.hstack([binary, morph, morph-binary])
        cv2.imshow('binary', full)
        if wait_key is not None:
            cv2.waitKey(wait_key)
    return morph, grey
