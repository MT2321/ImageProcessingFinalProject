import cv2 as cv
import cv2
import numpy as np
import os
from image_processing.BoxDetectorUtils import BoxDetectorUtils as bxu
from image_processing.BoxDetectorUtils import VideoManager, VideoManagerThumbs
SHOW = True

params = {
    "search_step_secs": 1.0,  # look for boxes each 1 second
    "duration_th": 1.0,  # secs
    "confidence_th": 0.3,
    "down_scale_factor": 0,
    "area_factor_low": 0.03,  # perecentage
    "area_factor_high": 0.42,  # perecentage
    "thc_low": 0,
    "thc_high": 50,
    "gauss_kernel": 5,
    "ero_dil_kernel_size": 3,  # Helps to connect segments and removes some noise
    "sigma": 2,
    "max_polig": 10,
    "group": True,
    "comp_tgt": "hsv_hist",
    "hist_long": 64,
    "IOU_TOL": 0.4,
    "HIST_TOL": 0.1,
}


class BroadBoxDetector:
    def __init__(
        self,
        do_extract=False,
        feed_id=None,
        source_id=None,
        clip_url=None,
        feed_name=None,
        start_time=None,
        default_params=params,
    ):

        self.process_history = {}
        # Dataset generations
        self.middle_box = None
        # FingerPrintStuff Flag
        self.do_extract = do_extract
        # FingerPrintStuff
        self.feed_id = feed_id
        self.source_id = source_id
        self.clip_url = clip_url
        self.feed_name = feed_name
        self.start_time = start_time

        # Load params
        for key, value in params.items():
            setattr(self, key, value)
        self.global_status = True

        # Preprocess_image constants
        self.ero_dil_kernel = np.ones(
            (self.ero_dil_kernel_size, self.ero_dil_kernel_size), np.uint8
        )

        # Hashers
        pHasher = cv.img_hash_PHash.create()
        # cHasher = cv.img_hash_ColorMomentHash.create()
        # bmHasher = cv.img_hash_BlockMeanHash.create()
        self.hashers = {
            "pHash": {"hasher": pHasher, "name": "pHash", "threshold": 15},
            # "cHash": {"hasher": cHasher, "name": "cHash", "threshold": 5},
            # "bmHash": {"hasher": bmHasher, "name": "bmHash", "threshold": 20},
        }

        # Results storage
        self.all_boxes = []
        self.fingerprints = []

    def search_boxes(self, video_path, thumbs_dir=None, middle_name_wb=None, middle_name_nb=None):

        self.middle_name_wb = middle_name_wb
        self.middle_name_nb = middle_name_nb

        self.video_path = video_path
        if thumbs_dir is not None:
            self.vm = VideoManagerThumbs(video_path, thumbs_dir)
        else:
            self.vm = VideoManager(video_path)

        self.all_boxes.clear()
        scanning = True
        current_sec_pos = 0.0

        while scanning:
            # print(f"\r {current_sec_pos}", end="")
            status, bgr_frame = self.vm.get_frame(current_sec_pos)
            if status is False:
                break
            hsv_frame = cv.cvtColor(bgr_frame, cv.COLOR_BGR2HSV)
            raw_boxes = self.detect_boxes(bgr_frame)

            if len(raw_boxes) > 0:
                found_boxes = self.add_metadata(
                    raw_boxes, bgr_frame, hsv_frame, current_sec_pos
                )
                found_boxes = self.delete_duplicates(found_boxes)
                self.all_boxes = self.all_boxes + found_boxes

            # Update end_times
            self.update_duration(bgr_frame, hsv_frame, current_sec_pos)

            current_sec_pos = current_sec_pos + self.search_step_secs

        for box in self.all_boxes:
            self.add_time_data(box)

        self.all_boxes = self.delete_too_short(self.all_boxes)

        for box in self.all_boxes:
            self.replace_for_middle_hist(box)
            self.calculate_confidence(box)

        self.plot_aid(show=SHOW)
        # print("Finished processing")
        # print(f"Boxes found {len(self.all_boxes)}")
        # print(f"time elapsed {self.clock}")
        return self.all_boxes, self.middle_box

    def detect_boxes(self, frame, find_zocalos=False):
        # import matplotlib.pyplot as plt
        pp_image = self.preprocess_image(frame)

        # Traditional box finding
        bounding_boxes_1 = self.find_all_boxes(pp_image, "box")

        # Second bbx search, improved lines
        h_image, v_image, lines_image = self.extract_vh_lines(frame)
        self.process_history["step 4 vh lines"] = lines_image.copy()

        lines_image_with_border = self.add_border(lines_image, 5, 4)
        self.process_history["step 5 artificial border"] = lines_image_with_border.copy(
        )

        bounding_boxes_2 = self.find_all_boxes(lines_image_with_border, "box")

        bounding_boxes_3 = []
        if find_zocalos:
            # Zocalo search
            pp_image_zocalo = self.preprocess_zocalo(frame)
            # plt.imshow(pp_image_zocalo)
            # plt.show()
            bounding_boxes_3 = self.find_all_boxes(pp_image_zocalo, "zocalo")

        self.raw_boxes = bounding_boxes_1 + bounding_boxes_2 + bounding_boxes_3

        boxes = self.merge_similar_boxes(
            bounding_boxes_1 + bounding_boxes_2 + bounding_boxes_3
        )
        # print(boxes)
        bounding_boxes = boxes
        return bounding_boxes

    def update_duration(self, bgr_next_frame, hsv_next_frame, secs_pos):

        for box in self.all_boxes:
            if not box["finished"]:
                if box["aspect_ratio"] > 7:
                    hue_hist, hsv_hist, bgr_hist = bxu.get_hue_hist(
                        bgr_next_frame, hsv_next_frame, box["box"]
                    )
                    if self.comp_tgt == "hsv_hist":
                        comp_tgt = hsv_hist
                    elif self.comp_tgt == "bgr_hist":
                        comp_tgt = bgr_hist
                    elif self.comp_tgt == "hist":
                        comp_tgt = hue_hist

                    if self.are_boxes_similar_cos_dis(box[self.comp_tgt], comp_tgt):
                        box["end_time"] = secs_pos
                    else:
                        box["finished"] = True

                else:
                    comp_hash = bxu.compute_hashes(
                        bgr_next_frame, box["box"], self.hashers
                    )
                    hash_simi = bxu.compare_hashes(
                        box["hashes"], comp_hash, self.hashers
                    )
                    if hash_simi["pHash"]["is_similar"]:
                        # if self.are_boxes_similar_cos_dis(box[self.comp_tgt], comp_tgt):
                        box["end_time"] = secs_pos
                    else:
                        box["finished"] = True

    def are_boxes_similar_cos_dis(self, hist_1, hist_2):
        hist_sim = bxu.is_similar(hist_1, hist_2, self.HIST_TOL)
        if hist_sim:
            return True
        else:
            return False

    def replace_for_middle_hist(self, box):
        ret, bgr_frame = self.vm.get_frame(box["middle_time"])
        hsv_frame = cv.cvtColor(bgr_frame, cv.COLOR_BGR2HSV)
        hue_hist, hsv_hist, bgr_hist = bxu.get_hue_hist(
            bgr_frame, hsv_frame, box=box["box"]
        )
        box["hist"] = hue_hist
        box["hsv_hist"] = hsv_hist
        box["bgr_hist"] = bgr_hist

    def add_time_data(self, box):
        box["duration_secs"] = np.round(
            (box["end_time"] - box["start_time"]), 2)
        box["middle_time"] = ((box["end_time"] + box["start_time"])) // 2

    def add_metadata(self, bounding_boxes, bgr_frame, hsv_frame, current_sec_pos):
        found_boxes = []

        for box in bounding_boxes:
            hue_hist, hsv_hist, bgr_hist = bxu.get_hue_hist(
                bgr_frame, hsv_frame, box=box
            )
            hashes = bxu.compute_hashes(bgr_frame, box, self.hashers)
            _, _, w, h = box
            found_boxes.append(
                {
                    "box": box,
                    "w": w,
                    "h": h,
                    "aspect_ratio": w / h,
                    "hist": hue_hist,
                    "hsv_hist": hsv_hist,
                    "bgr_hist": bgr_hist,
                    "start_time": current_sec_pos,
                    "end_time": current_sec_pos,
                    "middle_time": None,
                    "frame_rate": self.vm.get_fps(),
                    "duration_secs": 0,
                    "confidence": 0,
                    "finished": False,
                    "took": 0,
                    "storage_url": "",
                    "hashes": hashes,
                }
            )
        return found_boxes

    def preprocess_image(self, image):
        x = image.copy()
        x = cv.GaussianBlur(
            x,
            (self.gauss_kernel, self.gauss_kernel),
            sigmaX=self.sigma,
            sigmaY=self.sigma,
        )
        self.process_history["step 1 blur"] = x.copy()

        x = cv.Canny(x, self.thc_low, self.thc_high, L2gradient=False)
        self.process_history["step 2 canny"] = x.copy()

        # x = self.auto_canny(x)
        x = cv.dilate(x, self.ero_dil_kernel, iterations=1)
        self.process_history["step 3 dilate"] = x.copy()

        x = cv.erode(x, self.ero_dil_kernel, iterations=1)
        self.process_history["step 4 erode"] = x.copy()

        # ret, x = cv.threshold(x, 0, 255, cv.THRESH_BINARY)
        return x

    def add_border(self, gray_image, offset, thickness=1):
        h, w = gray_image.shape
        image = gray_image.copy()
        image[:, offset: offset + thickness] = 255  # Left side
        image[:, w - thickness - offset: w - offset] = 255  # Right side
        image[h - (thickness + offset):, :] = 255  # bottom
        image[0:thickness, :] = 255

        return image

    def extract_vh_lines(self, image):
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img_gray_gauss = cv.GaussianBlur(img_gray, (5, 5), 0)
        # apply automatic Canny edge detection using the computed median
        canny_image = self.auto_canny(img_gray_gauss)
        h, v, merged = bxu.split_image(canny_image)
        return h, v, merged

    def auto_canny(self, image):
        sigma = 0.33
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        canny_image = cv.Canny(image, lower, upper, 7)
        return canny_image

    def extract_box(
        self, countour, hierarchy, low_area_th, high_area_th, use_hierachy=True
    ):
        # Only admit contours that have child contours, no empty boxes
        box = None
        if hierarchy[2] == -1 and use_hierachy:
            return box
        else:
            epsilon = 0.05 * cv.arcLength(countour, True)
            cnt = cv.approxPolyDP(countour, epsilon, True)
            if 4 <= len(cnt) < self.max_polig:
                if cv.isContourConvex(cnt):
                    boundRect_temp = cv.boundingRect(cnt)
                    if low_area_th < bxu.rect_area(boundRect_temp) < high_area_th:
                        box = boundRect_temp
            return box

    def extract_zocalo(
        self,
        countour,
        low_area_th_z,
        high_area_th_z,
        center_x,
        radius_x_th,
        y_th,
        image_area,
    ):
        zocalo = None

        def get_box_center(bounding_box):
            return (
                (bounding_box[0] + bounding_box[2]) // 2,
                (bounding_box[1] + 2 * bounding_box[3]) // 2,
            )

        epsilon = 0.01 * cv.arcLength(countour, True)
        cnt = cv.approxPolyDP(countour, epsilon, True)
        if 4 <= len(cnt) < self.max_polig:
            boundRect_temp = cv.boundingRect(cnt)
            # print(boundRect_temp)
            bb_center_x, bb_center_y = get_box_center(boundRect_temp)
            is_centered = np.abs(bb_center_x - center_x) < radius_x_th
            # print("--------------")
            # print(f"y_coor {bb_center_y}")
            # print(f"y_th {y_th}")
            is_low = bb_center_y > y_th
            # print(f"x_center {bb_center_x}")
            # print(f"is_centered {is_centered}")
            # print(f"is_low {is_low}")
            if is_centered and is_low:
                if (
                    image_area * 0.10
                    < bxu.rect_area(boundRect_temp)
                    < image_area * 0.20
                ):
                    zocalo = boundRect_temp
            # area_prop = bxu.rect_area(zocalo) / image_area
            # print(f"ara_prop {area_prop}")
            # print("--------------")

        return zocalo

    def find_all_boxes(self, image, search_object, use_hierachy=True):
        "Receives already pp image"
        if search_object == "box":
            contours, hierarchies = cv.findContours(
                image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
            )
        elif search_object == "zocalo":
            contours, hierarchies = cv.findContours(
                image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
            )

        h, w = image.shape
        img_area = image.shape[0] * image.shape[1]
        low_area_th_box = self.area_factor_low * img_area
        high_area_th_box = self.area_factor_high * img_area

        bounding_boxes = []
        if len(contours) > 0:
            for cnt, hierarchy in zip(contours, hierarchies[0]):
                if search_object == "box":
                    box = self.extract_box(
                        cnt, hierarchy, low_area_th_box, high_area_th_box, use_hierachy
                    )
                    if box is not None:
                        bounding_boxes.append(box)

                elif search_object == "zocalo":
                    box = self.extract_zocalo(
                        cnt,
                        low_area_th_box,
                        high_area_th_box,
                        w // 2,
                        100,
                        h // 2,
                        img_area,
                    )
                    if box is not None:
                        bounding_boxes.append(box)
                else:
                    print("*box* and *zocalo* are the only valid argument")
        else:
            bounding_boxes = []
        return bounding_boxes

    def preprocess_zocalo(self, image, dilate_edges=1):
        pp_zocalo = self.preprocess_image(image)
        # Fill in the gaps
        pp_kernel_h = np.ones((dilate_edges, 1))
        pp_kernel_v = np.ones((1, dilate_edges))
        pp_h = cv.dilate(pp_zocalo, pp_kernel_h)
        pp_v = cv.dilate(pp_zocalo, pp_kernel_v)
        pp_zocalo = pp_h + pp_v

        # Extract Horizontal lines
        # Dilate as much as possible
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 1))
        # Remove very short lines
        eroded = cv.erode(pp_zocalo, horizontal_kernel, iterations=5)
        horizontal_kernel_dil = cv.getStructuringElement(
            cv.MORPH_RECT, (30, 1))
        # Strech the remaining
        dilated = cv.dilate(eroded, horizontal_kernel_dil, iterations=60)
        # Delete all lines that do no go across the frame
        long_horizontal_lines = cv.erode(
            dilated, horizontal_kernel_dil, iterations=100)

        # Add artifical border
        horizontal_lines_wborder = self.add_border(long_horizontal_lines, 5, 4)

        kernel = cv.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        horizontal_lines_wborder = cv.dilate(
            horizontal_lines_wborder, kernel, iterations=1)
        return horizontal_lines_wborder

    def calculate_confidence(self, box):
        box["confidence"] = np.round(
            box["confidence"] / box["duration_secs"], 2)

    # Geometry related methods
    def merge_similar_boxes(self, bounding_boxes):
        bounding_boxes = self.group_overlapping_boxes(bounding_boxes)
        bounding_boxes = self.compress_groups(bounding_boxes)
        return bounding_boxes

    def group_overlapping_boxes(self, bbxs):
        groups = []
        used = []
        for box1 in bbxs:
            if box1 not in used:
                used.append(box1)
                temp = [box1]
                for box2 in bbxs:
                    if box2 not in used and bxu.compute_iou(box1, box2) > self.IOU_TOL:
                        temp.append(box2)
                        used.append(box2)
                groups.append(temp)
        return groups

    def compress_groups(self, groups):
        groups = [self.compress_group(g) for g in groups]
        return groups

    def compress_group(self, bbxs):
        areas = [bxu.rect_area(box) for box in bbxs]
        indices = np.argsort(areas)
        smallest_box = indices[0]
        return bbxs[smallest_box]

    # Similarity methods

    # Noise Filtering
    def delete_duplicates(self, found_boxes):
        if len(self.all_boxes) == 0:
            new_found_boxes = found_boxes
        else:
            discarded_boxes = []
            for new_box in found_boxes:
                for old_box in self.all_boxes:
                    iou = bxu.compute_iou(new_box["box"], old_box["box"])
                    if iou > self.IOU_TOL:
                        hash_simi = bxu.compare_hashes(
                            new_box["hashes"], old_box["hashes"], self.hashers
                        )

                        is_zocalo = new_box["aspect_ratio"] > 7
                        is_hist_similar = bxu.is_similar(
                            new_box[self.comp_tgt],
                            old_box[self.comp_tgt],
                            tol=self.HIST_TOL,
                        )

                        if hash_simi["pHash"]["is_similar"] or (
                            is_zocalo and is_hist_similar
                        ):
                            if not old_box["finished"]:
                                old_box["confidence"] += 1
                            discarded_boxes.append(new_box)
                            break

            new_found_boxes = []
            for box in found_boxes:
                # print(f"len boxes {len(box)}")
                # print(f"len discarded {len(discarded_boxes)}")
                if box not in discarded_boxes:
                    new_found_boxes.append(box)
        return new_found_boxes

    def delete_too_short(self, found_boxes):
        found_boxes = list(
            filter(lambda item: item["duration_secs"]
                   >= self.duration_th, found_boxes)
        )
        return found_boxes

    def delete_low_confidence(self, found_boxes):
        found_boxes = list(
            filter(lambda item: item["confidence"] >
                   self.confidence_th, found_boxes)
        )
        return found_boxes

    def plot_aid(self, show=False):
        import matplotlib.pyplot as plt

        # from matplotlib import cm

        blue = (255, 255, 0)
        for counter, box in enumerate(self.all_boxes):
            # print(
            #     f"duration {box['duration_secs']} || confidence  {box['confidence']} || ar  {box['aspect_ratio']}"
            # )
            f, ax = plt.subplots(1, 5, figsize=(30, 4))
            for i in range(5):
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)

            ret, beg_frame_m1 = self.vm.get_frame(
                max(box["start_time"] - self.search_step_secs, 0)
            )
            beg_frame_m1 = bxu.draw_boxes(
                cv.cvtColor(beg_frame_m1, cv.COLOR_BGR2RGB), [box["box"]]
            )

            ret, beg_frame = self.vm.get_frame(box["start_time"])
            beg_frame = bxu.draw_boxes(
                cv.cvtColor(beg_frame, cv.COLOR_BGR2RGB), [box["box"]], color=blue
            )
            ret, end_frame = self.vm.get_frame(box["end_time"])
            end_frame = bxu.draw_boxes(
                cv.cvtColor(end_frame, cv.COLOR_BGR2RGB), [box["box"]], color=blue
            )

            ret_p1, end_frame_p1 = self.vm.get_frame(
                box["end_time"] + self.search_step_secs
            )
            if ret_p1:
                end_frame_p1 = bxu.draw_boxes(
                    cv.cvtColor(end_frame_p1, cv.COLOR_BGR2RGB), [box["box"]]
                )

            ax[0].set_title(f"{self.search_step_secs} segs antes")
            ax[0].imshow(beg_frame_m1)
            ax[1].set_title(f'Comienzo {box["start_time"]}')
            ax[1].imshow(beg_frame)
            ax[2].set_title(
                f'Final {box["end_time"]} dur: {box["duration_secs"]} segs '
            )
            ax[2].imshow(end_frame)
            if ret_p1 is True:
                ax[3].set_title(f"{self.search_step_secs} segs despues")
                ax[3].imshow(end_frame_p1)
            else:
                ax[3].set_title("Final")
                ax[3].imshow(end_frame)
            ax[4].set_title(f"Middle Frame | confidence {box['confidence']}")
            ret, middle_frame = self.vm.get_frame(box["middle_time"])
            middle_rgb_nbox = middle_frame.copy()
            middle_frame = bxu.draw_boxes(
                middle_frame, [box["box"]], color=(255, 70, 0)
            )
            middle_rgb_wbox = cv.cvtColor(middle_frame, cv.COLOR_BGR2RGB)
            middle_rgb_nbox = cv.cvtColor(middle_rgb_nbox, cv.COLOR_BGR2RGB)
            ax[4].imshow(middle_rgb_wbox)
            if self.middle_name_nb is not None and box["aspect_ratio"] > 7:
                # print("saving image")
                plt.imsave(self.middle_name_nb, middle_rgb_nbox)
                plt.imsave(self.middle_name_wb, middle_rgb_wbox)
                self.middle_box = box["box"]
            else:
                self.middle_box = None
            video_name = self.video_path[0: self.video_path.find(".")]
            # print(box["box"])
            img_name = f"-{counter}-d-{box['duration_secs']}secs-c{box['confidence']}-{video_name}.png"
            img_name = img_name.replace("/", "")
            plt.savefig(img_name)
            os.remove(img_name)
            if show:
                plt.show()
            plt.close(f)

        return

    def plot_boxes(self, frame, boxes, show_history=False):
        import matplotlib.pyplot as plt

        # from matplotlib import cm
        colors = [
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255)
        ]
        image_with_boxes = frame.copy()
        image_with_raw_boxes = frame.copy()
        if not show_history:
            f, ax = plt.subplots(1, 1, figsize=(10, 50))
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        else:
            f, axes = plt.subplots(
                len(self.process_history)+1+1, 1, figsize=(30, 160))
            for ax, step in zip(axes, self.process_history.keys()):
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                ax.set_title(step,  fontsize=160)
                ax.imshow(self.process_history[step])

        for counter, box in enumerate(self.raw_boxes):
            image_with_raw_boxes = bxu.draw_boxes(
                image_with_raw_boxes,
                [box],
                color=colors[counter % len(colors)]
            )

        for counter, box in enumerate(boxes):
            image_with_boxes = bxu.draw_boxes(
                image_with_boxes,
                [box],
                color=colors[counter % len(colors)]
            )
        self.process_history["final_result"] = image_with_boxes
        if not show_history:
            ax.set_title("Frame with boxes")
            ax.imshow(image_with_boxes)
        else:
            axes[-2].set_title("Frame with raw boxes", fontsize=200)
            axes[-2].imshow(image_with_raw_boxes)
            axes[-1].set_title("Frame with boxes", fontsize=200)
            axes[-1].imshow(image_with_boxes)

        # img_name = img_name.replace("/", "")
        # plt.savefig(img_name)
        # os.remove(img_name)
        plt.show()
        plt.close(f)
        print(self.process_history)
        return self.process_history
