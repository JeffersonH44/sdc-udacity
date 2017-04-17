from lesson_functions import *
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import pickle
import params
import itertools
import tensorflow as tf

from scipy.ndimage.measurements import label
from keras.backend.tensorflow_backend import set_session
from ssd_v2 import SSD300v2
from ssd_utils import BBoxUtility

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))


def create_windows(pyramid, image_size, xy_overlap=(0.6, 0.6)):
    output = []
    for w_size, y_lims in pyramid:
        windows = slide_window(image_size, x_start_stop=[None, None], y_start_stop=y_lims,
                               xy_window=w_size, xy_overlap=xy_overlap)
        output.extend(windows)
    return output


class VehiclePrediction:
    def __init__(self, heat_map_frame=10, confidence=0.3):
        self.confidence = confidence

        # update heat map
        self.heat_map_frame = heat_map_frame
        self.current_frame = 0
        self.heat_map = None
        self.threshold = heat_map_frame // 2
        self.label_map = None

        self.voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                            'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                            'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                            'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        NUM_CLASSES = len(self.voc_classes) + 1

        input_shape = (300, 300, 3)
        self.model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
        self.model.load_weights('weights_SSD300.hdf5', by_name=True)
        self.bbox_util = BBoxUtility(NUM_CLASSES)

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
        self.heat_map[self.heat_map <= self.threshold] = 0
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

    def predict(self, img):
        if self.current_frame == 0:
            self.heat_map = np.zeros_like(img[:, :, 0]).astype(np.float)

        resize_img = np.array([cv2.resize(img, (300, 300))])
        output = self.model.predict(resize_img)
        results = self.bbox_util.detection_out(output)

        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]

        top_conf = det_conf[top_indices]
        # top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        hot_windows = []

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            hot_windows.append([(xmin, ymin), (xmax, ymax)])

        self.add_heat(hot_windows)
        if self.current_frame == self.heat_map_frame:
            self.current_frame = 0
            self.apply_threshold()
            update = True
        else:
            self.current_frame += 1
            update = False

        img = self.draw_labeled_bboxes(img, update=update)
        # window_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
        # if show:
        #    plt.imshow(img)
        #    plt.show()
        return img

    def predict_prob(self, img):
        resize_img = np.array([cv2.resize(img, (300, 300))])
        output = self.model.predict(resize_img)
        results = self.bbox_util.detection_out(output)

        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            pt1, pt2 = (xmin, ymin), (xmax, ymax)

            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = self.voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            # color = colors[label]
            img = cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, display_txt, (pt1[0] - 10, pt1[1]), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def show_stats(self, img):
        if self.current_frame == 0:
            self.heat_map = np.zeros_like(img[:, :, 0]).astype(np.float)

        stats_image = np.copy(img)
        pred_image = np.copy(img)
        resize_img = np.array([cv2.resize(img, (300, 300))])
        output = self.model.predict(resize_img)
        results = self.bbox_util.detection_out(output)

        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        hot_windows = []

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            pt1, pt2 = (xmin, ymin), (xmax, ymax)

            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = self.voc_classes[label - 1]
            if label_name != 'Car':
                continue
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            # color = colors[label]
            stats_image = cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            stats_image = cv2.putText(img, display_txt, (pt1[0] - 10, pt1[1]), font, 0.5, (255, 255, 255), 2,
                                      cv2.LINE_AA)
            hot_windows.append((pt1, pt2))

        self.add_heat(hot_windows)
        if self.current_frame == self.heat_map_frame:
            self.current_frame = 0
            self.apply_threshold()
            update = True
        else:
            self.current_frame += 1
            update = False

        pred_image = self.draw_labeled_bboxes(pred_image, update=update)
        # window_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
        # if show:
        #    plt.imshow(img)
        #    plt.show()
        return np.concatenate((pred_image, stats_image), axis=1)

    def produce_video(self, input_path, output_path, get_probs=False):
        self.current_frame = 0
        pred = self.show_stats  # self.predict_prob if get_probs else self.predict
        clip2 = VideoFileClip(input_path)
        clip = clip2.fl_image(pred)
        clip.write_videofile(output_path, audio=False)

source_video = 'project_video.mp4'
output_video = './output_images/prob_output.mp4'
vp = VehiclePrediction(heat_map_frame=15, confidence=0.07)
vp.produce_video(source_video, output_video, get_probs=True)
