class Configuration:

    def __init__(self):

        self.video_path = "data/input/videos/2.mp4"
        self.output_json_path = "dummy.json"
        self.input_frame_blur_kernel_size =  None# A number or None not zero.
        self.resize_input_image_height = 800
        self.morphology_size = (7, 7)
        self.grad_min = 20

        self.log_warnings = True
        self.log_errors = True
        self.log_info = False
        self.min_object_area = 5