from lesson_functions import *
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import pickle
import params
import itertools

from scipy.ndimage.measurements import label


class VehiclePrediction:
    def __init__(self, heat_map_frame=7, threshold=6):
        # load default classifier
        self.clf = pickle.load(open('classifier.p', 'rb'))
        # load default normalizer
        self.scaler = pickle.load(open('norm.p', 'rb'))
        # update heat map
        self.heat_map_frame = heat_map_frame
        self.current_frame = 0
        self.heat_map = None
        self.threshold = threshold
        self.label_map = None

        # for image processing
        self.orient = params.orient
        self.pix_per_cell = params.pix_per_cell
        self.cell_per_block = params.cell_per_block
        self.spatial_size = params.spatial_size
        self.hist_bins = params.hist_bins
        self.hog_channel = params.hog_channel
        self.color_space = params.color_space

    def add_heat(self, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        # return heatmap  # Iterate through list of bboxes

    def apply_threshold(self):
        # Zero out pixels below the threshold
        # heat_map = np.copy(self.heat_map)
        self.heat_map[self.heat_map <= self.threshold] = 0

        # subplots(2, [heat_map, self.heat_map], ["Original heat map", "heat map with threshold"], cmap='hot')
        # Return thresholded map

    def draw_labeled_bboxes(self, img, update=False):
        # Iterate through all detected cars
        if update:
            self.label_map = label(self.heat_map)
        if self.label_map is None:
            return img

        for car_number in range(1, self.label_map[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (self.label_map[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    def image_pipeline(self, img, show=True):
        draw_img = np.copy(img)

        if self.current_frame == 0:
            self.heat_map = np.zeros_like(img[:, :, 0]).astype(np.float)

        pyramid = [((64, 64), [400, 500]),
                   ((96, 96), [400, 550]),
                   ((128, 128), [450, 550]),
                   ]
        windows = create_windows(pyramid, img.shape[:2], xy_overlap=(0.5, 0.5))
        hot_windows = search_windows(img, windows, clf=self.clf, scaler=self.scaler, color_space=self.color_space,
                                     spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                     orient=self.orient, pix_per_cell=self.pix_per_cell,
                                     cell_per_block=self.cell_per_block,
                                     hog_channel=self.hog_channel)

        self.add_heat(hot_windows)
        if self.current_frame == self.heat_map_frame:
            #tst_img = np.copy(img)
            #tst_img = draw_boxes(tst_img, windows)
            #subplots(2, [img, tst_img], ["Original image", "Sliding window scheme"])
            self.current_frame = 0
            self.apply_threshold()
            update = True
        else:
            self.current_frame += 1
            update = False

        draw_img = self.draw_labeled_bboxes(draw_img, update=update)
        # window_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
        if show:
            plt.imshow(draw_img)
            plt.show()
        return draw_img

    def produce_video(self, input_path, output_path, show=False):
        self.current_frame = 0
        clip2 = VideoFileClip(input_path)
        clip = clip2.fl_image(lambda x: self.image_pipeline(x, show=show))
        clip.write_videofile(output_path, audio=False)


source_video = 'test_video.mp4'
output_video = './output_images/output_vid.mp4'
vp = VehiclePrediction(heat_map_frame=8, threshold=5)
vp.produce_video(source_video, output_video, show=True)