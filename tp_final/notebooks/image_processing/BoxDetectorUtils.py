import cv2
import cv2 as cv
import numpy as np
import time
from scipy.spatial import distance
from datetime import datetime
from pathlib import Path
import os
from subprocess import call

# from resources.boxes.models import FrameHistogram


def ffp_get_index(feed_id):
    return "ffp_" + str(feed_id) + "_" + datetime.today().strftime("%Y-%m-%d")


def gm_detected_box_get_index(feed_id):
    return "gm_detected_box_" + str(feed_id) + "_" + datetime.today().strftime("%Y-%m-%d")


class VideoManager:
    def __init__(self, video_path):
        self.video = cv.VideoCapture(video_path)
        # self.video.set(cv.CAP_PROP_CONVERT_RGB, True) # para el futuro
        self.fps = self.video.get(cv.CAP_PROP_FPS)
        self.cached_frames = {}

    def pre_load_frames(self, secs_time_step):
        self.cached_frames.clear()
        i = 0
        status = True
        self.secs_time_step = secs_time_step
        self.preloaded = True
        while status:
            status, frame = self.get_frame(secs_time_step * i)
            i = i + secs_time_step

    def get_frame(self, sec_pos):
        ms_pos = self.sec2ms(sec_pos)
        self.video.set(cv.CAP_PROP_POS_MSEC, ms_pos)
        ret, frame = self.video.read()

        return ret, frame

    def reset(self):
        self.cached_frames = {}

    @staticmethod
    def sec2ms(sec):
        ms = np.round(sec * 1000, 2)
        return ms

    @staticmethod
    def ms2sec(ms):
        secs = np.round(ms / 1000, 2)
        return secs

    def get_fps(self):
        return self.fps


class VideoManagerThumbs:
    def __init__(self, video_path, thumbs_dir, digits=4, duration=None):
        self.video_path = Path(video_path)
        self.thumbs_dir = Path(thumbs_dir).joinpath(self.video_path.stem)
        self.thumbs_dir.mkdir(exist_ok=True)
        self.digits = 4
        self.duration = duration
        self.extract_thumbs()

    def extract_thumbs(self):
        cmd = ["/usr/bin/ffmpeg", "-hide_banner",
               "-y", "-i", str(self.video_path)]
        cmd += ["-f", "image2"]
        cmd += [
            "-loglevel",
            "error",
            "-q:v",
            "12",
            "-filter:v",
            (
                "fps=1/1,"
                "scale='ceil(oh*dar/2)*2':'min({max_height},ceil(ih/2)*2)',"
                "setsar=1/1"
            ).format(max_height=480),
            os.path.join(str(self.thumbs_dir), "%04d.jpg"),
        ]

        call(cmd)

        # if command_run != 0:
        #     clean_cmd = " ".join(cmd)
        #     print(f"Error extracting thumbs {clean_cmd}")
        #     raise NoThumbsError

    def get_frame(self, sec_pos):
        frame_path = str(self.thumbs_dir.joinpath(
            self.secs2format(sec_pos))) + ".jpg"
        frame = cv.imread(frame_path)
        if frame is not None:
            ret = True
        else:
            ret = False

        return ret, frame

    def secs2format(self, sec_pos):
        s = str(int(sec_pos))
        s = (self.digits - len(s)) * "0" + s
        return s

    def get_fps(self):
        return 0


class Profiler:
    def __init__(self, name):
        self.start_time = 0
        self.end_time = 0
        self.data = []
        self.name = name

    def start(self):
        self.start_time = time.time()

    def stop(self, save_data=True):
        self.stop_time = time.time()
        if save_data:
            self.save_data()

    def save_data(self):
        self.data.append(self.get_elapsed())

    def get_elapsed(self):
        return self.stop_time - self.start_time

    def avg_elapsed(self):
        return np.round(np.mean(self.data), 2)

    def get_max(self):
        return np.round(np.max(self.data), 2)

    def get_acc(self):
        return np.round(np.sum(self.data), 2)

    def reset(self):
        self.start_time = 0
        self.end_time = 0
        self.data = []

    def __str__(self):
        if len(self.data) > 0:
            return f"{self.name} \t {self.get_max()} \t\t {self.avg_elapsed()} \t\t {self.get_acc()}"
        else:
            return f"{self.name} \t EMPTY \t\t EMPTY \t\t EMPTY"


class BoxDetectorUtils:
    """
    Helper methods for Box Detector

    """

    def __init__(self):
        pass

    @staticmethod
    def down_scale(image, down_scale_factor):
        """
        Down scales images by a factor of 2*down_scale_factor


        Parameters
        ----------
        image : cv.Mat
            input image to downscale
        down_scale_factor : int, optional
            number of times to halve image's resolution, by default 1

        Returns
        -------
        cv.Mat
            Downscaled image
        """
        temp = image.copy()
        for i in range(down_scale_factor):
            temp = cv.pyrDown(temp)  # downscale by 2
        return temp

    @staticmethod
    def crop_frame(frame, box):
        """crop_frame [summary]

        Parameters
        ----------
        frame : [type]
            [description]
        box : array
            [description]
        """
        x0, y0, w, h = box
        crop = frame[y0: y0 + h, x0: x0 + w]
        return crop

    @staticmethod
    def reformat_box(box):
        """
        Changes OpenCV box representation for convenience
        A box with [x0, y0, w, h] is received. Where
        x0 and y0 represent the upper left corner
        and w and h specify width and height.


        Parameters
        ----------
        box : array
            Rectangular bounding box reprensetation
            A box with [x0, y0, w, h]

        Returns
        -------
        box : [array of tuples]
            Reformated box containing the upper left
            corner and the lower right corner
            [(x0, y0), (x0 + w, y0 + h)]
            [(x0,y0), (x1,y1)]
        """
        x0, y0, w, h = box
        ref_box = [(x0, y0), (x0 + w, y0 + h)]
        return ref_box

    @staticmethod
    def rect_area(box):
        """
        Computes the area of an OpenCV box representation.

        Parameters
        ----------
        box : array
            [x0, y0, w, h]
            where x0,y0 are the coordinates of the upper right corner
            w and h specify width and height

        Returns
        -------
        area: int
            area of the box in pixels squared.
        """

        _, _, w, h = box
        area = np.abs(w * h)
        return area

    @staticmethod
    def rescale_box(box, up_factor):
        """
        Map coordinates from low resolution space to a higher resolution space

        Parameters
        ----------
        box : array
            [x0, y0, w, h]
            where x0,y0 are the coordinates of the upper right corner
            w and h specify width and height

        up_factor : int
            reescaling factor, usually matches the

        Use Cases
        ---------
        Scale box coordinates calculated in a low resolution space.
        After processing a downscaled image its useful to rescale the obtained box
        in order to match the original resolution.

        Returns
        -------
        reescaled_box : array
            [2*x0*up_factor, 2*y0*up_factor, 2*w*up_factor, 2*h*up_factor]

        """

        if up_factor != 0:
            x0, y0, w, h = box
            up_factor = 2 * up_factor
            return [x0 * up_factor, y0 * up_factor, w * up_factor, h * up_factor]
        else:
            return box

    @staticmethod
    def compute_iou(box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        box1 : dict
            Keys: {'x0', 'y0', 'w', 'h'}
            The (x0, y0) position is at the top left corner,
            the (w, h) width and height
        box2 : dict
            Keys: {'x0', 'y0', 'w', 'h'}
            The (x0, y0) position is at the top left corner,
            the (w, h) width and height

        Returns
        -------
        float
            in [0, 1]
        """
        # determine the coordinates of the intersection rectangle
        (b1x0, b1y0), (b1x1, b1y1) = BoxDetectorUtils.reformat_box(box1)
        (b2x0, b2y0), (b2x1, b2y1) = BoxDetectorUtils.reformat_box(box2)
        # determine the coordinates of the intersection rectangle
        x_left = max(b1x0, b2x0)
        y_top = max(b1y0, b2y0)
        x_right = min(b1x1, b2x1)
        y_bottom = min(b1y1, b2y1)

        # No intersect
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = max((x_right - x_left), 0) * \
            max((y_bottom - y_top), 0)
        # compute the area of both boxes
        _, _, w1, h1 = box1
        box1_area = w1 * h1

        _, _, w2, h2 = box2
        box2_area = w2 * h2

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / \
            float(box1_area + box2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    @staticmethod
    def get_bgr_hist(bgr_image, mask=None, bins=64):
        B_hist = np.reshape(cv.calcHist(
            [bgr_image], [0], mask, [bins], [0, 180]), -1)
        G_hist = np.reshape(cv.calcHist(
            [bgr_image], [1], mask, [bins], [0, 180]), -1)
        R_hist = np.reshape(cv.calcHist(
            [bgr_image], [2], mask, [bins], [0, 180]), -1)
        bgr_hist = np.hstack((B_hist, G_hist, R_hist))
        bgr_hist = (bgr_hist/np.max(bgr_hist))*0.25
        return bgr_hist

    # Similarity measurements
    @staticmethod
    def get_hue_hist(bgr_image, hsv_image, box=None, bins=64):
        mask = None
        if box is not None:
            mask = np.zeros(bgr_image.shape[:2], np.uint8)
            x0, y0, w, h = box
            mask[y0: y0 + h, x0: x0 + w] = 255
        else:
            mask = None

        B_hist = np.reshape(cv.calcHist(
            [bgr_image], [0], mask, [bins], [0, 180]), -1)
        G_hist = np.reshape(cv.calcHist(
            [bgr_image], [1], mask, [bins], [0, 180]), -1)
        R_hist = np.reshape(cv.calcHist(
            [bgr_image], [2], mask, [bins], [0, 180]), -1)
        bgr_hist = np.hstack((B_hist, G_hist, R_hist))

        hue_hist = np.reshape(cv.calcHist(
            [hsv_image], [0], mask, [bins], [0, 180]), -1)
        sat_hist = np.reshape(cv.calcHist(
            [hsv_image], [1], mask, [bins], [0, 180]), -1)
        value_hist = np.reshape(
            cv.calcHist([hsv_image], [2], mask, [bins], [0, 180]), -1
        )

        hsv_hist = np.hstack((hue_hist, value_hist, sat_hist))
        return hue_hist, hsv_hist, bgr_hist

    @staticmethod
    def is_similar(item1, item2, tol):
        """is_similar
        compare two vector elements using cosine distance
        If they are close enough returns True

        Parameters
        ----------
        item1 : array
            1D vector
        item2 : array
            1D vector
        tol : float
            maximum cosine distance allowed. Small numbers (1e3<) will check for very similar vectors

        Returns
        -------
        bool
            True if the cosine distance is below the tolerance threshold. False otherwise
        """
        d = distance.cosine(item1, item2)
        return d < tol

    def draw_box(image, box, color=(255, 0, 0)):
        """
        Overlays an image with a given box

        Parameters
        ----------
        image: array_like

        box: array of tuples.
                start:upper-left corner
                end: lower right-corner

                start    end
            [ (x0,y0), (x1,y1) ]

        """
        result = image.copy()
        cv.rectangle(result, box[0], box[1], color, 5)
        return result

    def draw_boxes(image, bbxs, color=(255, 0, 0)):
        """draw_boxes [summary]
        Reformatting is dine
        Parameters
        ----------
        image : [type]
            [description]
        bbxs : [type]
            [description]
        color : tuple, optional
            [description], by default (255, 0, 0)

        Returns
        -------
        [type]
            [description]
        """
        res = image.copy()
        if len(bbxs) >= 1:
            for bb in bbxs:
                res = BoxDetectorUtils.draw_box(
                    res, BoxDetectorUtils.reformat_box(bb), color=color
                )
        return res

    def split_image(image):
        """
        Receives a Ca edge detected image
        """
        # Obtain horizontal lines mask
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 1))
        horizontal_mask = cv.morphologyEx(
            image, cv.MORPH_OPEN, horizontal_kernel, iterations=2
        )
        jhl = horizontal_mask.copy()
        horizontal_mask = cv.dilate(
            horizontal_mask, horizontal_kernel, iterations=10)
        jhlx = horizontal_mask.copy()
        # Obtain vertical lines mask
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 30))
        vertical_mask = cv.morphologyEx(
            image, cv.MORPH_OPEN, vertical_kernel, iterations=2
        )
        jvl = vertical_mask.copy()
        vertical_mask = cv.dilate(
            vertical_mask, vertical_kernel, iterations=10)
        jvlx = vertical_mask.copy()
        # Combine vertical and horizontal lines
        merged = cv.bitwise_or(vertical_mask, horizontal_mask)

        return horizontal_mask, vertical_mask, merged, jhl, jhlx, jvl, jvlx

    def flatten_list(array):
        flat_list = [item for sublist in array for item in sublist]
        return flat_list

    @staticmethod
    def compute_hashes(image, box, hashers):
        cropped_image = BoxDetectorUtils.crop_frame(image, box)
        hashes = {}
        for hash_name, hasher in hashers.items():
            hashes[hash_name] = hasher["hasher"].compute(cropped_image)
        return hashes

    @staticmethod
    def compare_hashes(hash_1, hash_2, hashers):
        similarities = {}
        for hash_name, hasher in hashers.items():
            similarities[hash_name] = {
                "similarity": hasher["hasher"].compare(
                    hash_1[hash_name], hash_2[hash_name]
                )
            }
            similarities[hash_name]["is_similar"] = (
                similarities[hash_name]["similarity"] < hasher["threshold"]
            )
        return similarities

