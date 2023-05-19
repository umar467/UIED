class Configuration:

    def __init__(self):
        self.min_text_confidence = 0.7
        self.ssim_threshold = 0.5
        self.new_UI_layout_change_ratio = 0.8
        self.EAST_PATH = "data/model_weights/east_icdar2015_resnet_v1_50_rbox"
        self.maximum_SIFT_points_per_frame = 3000
        self.maximum_component_ratio = 10
        self.maximum_width_ratio = 40
        self.maximum_height_ratio = 50

        self.input_video = "data/input/videos/1.mp4"
        # self.output_json_folder = "json/"

        self.progress_callback = None
        self.server = False
        self.json_explicit_path = ''

        self.input_frame_blur_kernel_size = None  # A number or None not zero.
        self.resize_input_image_height = 800
        self.morphology_size = (7, 7)
        self.minimum_gradient_difference = 20
        self.frame_buffer_size = 10
        self.binary_dilation_iterations = 2
        self.minimum_component_height = 10
        self.minimum_component_width = 10
        self.maximum_component_height = 80
        self.maximum_component_width = 80
        self.minimum_component_area = 15
        self.log_warnings = True
        self.log_errors = True
        self.log_info = False
        self.min_object_area = 5

        self.COLOR = {'Button': (0, 255, 0), 'Compo': (0, 255, 0), 'CheckBox': (0, 0, 255),
                      'Chronometer': (255, 166, 166),
                      'EditText': (255, 166, 0),
                      'ImageButton': (77, 77, 255), 'ImageView': (255, 0, 166), 'ProgressBar': (166, 0, 255),
                      'RadioButton': (166, 166, 166),
                      'RatingBar': (0, 166, 255), 'SeekBar': (0, 166, 10), 'Spinner': (50, 21, 255),
                      'Switch': (80, 166, 66), 'ToggleButton': (0, 66, 80), 'VideoView': (88, 66, 0),
                      'TextView': (169, 255, 0),

                      'Text': (169, 255, 0), 'Non-Text': (255, 0, 166),

                      'Noise': (6, 6, 255), 'Non-Noise': (6, 255, 6),

                      'Image': (255, 6, 6), 'Non-Image': (6, 6, 255)}

        self.THRESHOLD_REC_MIN_EVENNESS = 0.7
        self.THRESHOLD_REC_MAX_DENT_RATIO = 0.25
        self.THRESHOLD_LINE_THICKNESS = 8
        self.THRESHOLD_LINE_MIN_LENGTH = 0.95
        self.THRESHOLD_COMPO_MAX_SCALE = (
        0.25, 0.98)  # (120/800, 422.5/450) maximum height and width ratio for a atomic compo (button)
        self.THRESHOLD_TEXT_MAX_WORD_GAP = 10
        self.THRESHOLD_TEXT_MAX_HEIGHT = 0.04  # 40/800 maximum height of text
        self.THRESHOLD_TOP_BOTTOM_BAR = (0.045, 0.94)  # (36/800, 752/800) height ratio of top and bottom bar
        self.THRESHOLD_BLOCK_MIN_HEIGHT = 0.03  # 24/800
