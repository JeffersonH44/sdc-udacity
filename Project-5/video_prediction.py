from lesson_functions import *
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import pickle
import params
import itertools

source_video = 'test_video.mp4'
output_video = './output_images/output.mp4'


def create_windows(pyramid, image_size):
    output = []
    for w_size, y_lims in pyramid:
        windows = slide_window(image_size, x_start_stop=[None, None], y_start_stop=y_lims,
                               xy_window=w_size, xy_overlap=(0.5, 0.5))
        output.extend(windows)
    return output


class VehiclePrediction:
    def __init__(self, heat_map_frame=8):
        # load default classifier
        self.clf = pickle.load(open('classifier.p', 'rb'))
        # load default normalizer
        self.scaler = pickle.load(open('norm.p', 'rb'))
        # update heat map
        self.heat_map_frame = heat_map_frame

        # for image processing
        self.y_start = params.y_start
        self.y_stop = params.y_stop
        self.orient = params.orient
        self.pix_per_cell = params.pix_per_cell
        self.cell_per_block = params.cell_per_block
        self.spatial_size = params.spatial_size
        self.hist_bins = params.hist_bins
        self.hog_channel = params.hog_channel
        self.color_space = params.color_space
        self.scale = params.scale

    def image_pipeline(self, img, show=True):
        pyramid = [((64, 64), [300, 600]),
                   ((96, 96), [300, 600]),
                   ((128, 128), [300, 600]),
                   ]
        windows = create_windows(pyramid, img.shape[:2])

        draw_img = np.copy(img)

        hot_windows = search_windows(img, windows, clf=self.clf, scaler=self.scaler, color_space=self.color_space,
                                     spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                     orient=self.orient, pix_per_cell=self.pix_per_cell,
                                     cell_per_block=self.cell_per_block,
                                     hog_channel=self.hog_channel)

        window_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
        if show:
            plt.imshow(window_img)
            plt.show()
        return window_img

    def produce_video(self, input_path, output_path, video_stats=False):
        clip2 = VideoFileClip(input_path)
        clip = clip2.fl_image(lambda x: self.image_pipeline(x))
        clip.write_videofile(output_path, audio=False)


vp = VehiclePrediction()

for i in range(1, 7):
    image = mpimg.imread("./test_images/test" + str(i) + ".jpg")
    vp.image_pipeline(image)
# vp.produce_video(source_video, output_video)
