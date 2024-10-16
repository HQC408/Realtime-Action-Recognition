#!/usr/bin/env python
# coding: utf-8

'''

Đọc hình ảnh huấn luyện dựa trên `valid_images.txt` và sau đó phát hiện skeleton.
    
Trong mỗi hình ảnh, chỉ có 1 người thực hiện một loại hành động.
Mỗi hình ảnh được đặt tên là 00001.jpg, 00002.jpg, ...

Một ví dụ về nội dung của valid_images.txt được hiển thị dưới đây:
    
    jump_03-12-09-18-26-176
    58 680

    jump_03-13-11-27-50-720
    65 393

    kick_03-02-12-36-05-185
    54 62
    75 84

Hai chỉ số (như `56 680` trong ví dụ `jump` đầu tiên)
đại diện cho chỉ số bắt đầu và kết thúc của một hành động cụ thể.

Input:
    SRC_IMAGES_DESCRIPTION_TXT
    SRC_IMAGES_FOLDER
    
Output:
    DST_IMAGES_INFO_TXT
    DST_DETECTED_SKELETONS_FOLDER
    DST_VIZ_IMGS_FOLDER
'''

import cv2
import yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_skeletons_io import ReadValidImagesAndActionTypesByTxt
    import utils.lib_commons as lib_commons


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings


cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s1_get_skeletons_from_training_imgs.py"]

IMG_FILENAME_FORMAT = cfg_all["image_filename_format"]
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]


# Input
if True:
    SRC_IMAGES_DESCRIPTION_TXT = par(cfg["input"]["images_description_txt"])
    SRC_IMAGES_FOLDER = par(cfg["input"]["images_folder"])

# Output
if True:
    # Tệp này sẽ lưu thông tin hình ảnh, như chỉ số, nhãn hành động, tên tệp, v.v.
    DST_IMAGES_INFO_TXT = par(cfg["output"]["images_info_txt"])

    # Mỗi tệp txt sẽ lưu skeleton của mỗi hình ảnh
    DST_DETECTED_SKELETONS_FOLDER = par(
        cfg["output"]["detected_skeletons_folder"])

    # Mỗi hình ảnh được vẽ với skeleton đã phát hiện
    DST_VIZ_IMGS_FOLDER = par(cfg["output"]["viz_imgs_folders"])

# Openpose
if True:
    OPENPOSE_MODEL = cfg["openpose"]["model"]
    OPENPOSE_IMG_SIZE = cfg["openpose"]["img_size"]

# -- Functions


class ImageDisplayer(object):
    ''' Một bao gồm đơn giản của việc sử dụng cv2.imshow để hiển thị hình ảnh '''

    def __init__(self):
        self._window_name = "cv2_display_window"
        cv2.namedWindow(self._window_name)

    def display(self, image, wait_key_ms=1):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):
        cv2.destroyWindow(self._window_name)

IMG_FILENAME_FORMAT = "{:05d}.png"  # Cập nhật định dạng tên tệp

if __name__ == "__main__":
    # In đường dẫn đầu vào để kiểm tra
    print(f"SRC_IMAGES_FOLDER: {SRC_IMAGES_FOLDER}")
    print(f"SRC_IMAGES_DESCRIPTION_TXT: {SRC_IMAGES_DESCRIPTION_TXT}")

    # -- Detector
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    multiperson_tracker = Tracker()

    # -- Image reader and displayer
    images_loader = ReadValidImagesAndActionTypesByTxt(
        img_folder=SRC_IMAGES_FOLDER,
        valid_imgs_txt=SRC_IMAGES_DESCRIPTION_TXT,
        img_filename_format=IMG_FILENAME_FORMAT)
    images_loader.save_images_info(filepath=DST_IMAGES_INFO_TXT)
    img_displayer = ImageDisplayer()

    # -- Khởi tạo đường dẫn đầu ra
    os.makedirs(os.path.dirname(DST_IMAGES_INFO_TXT), exist_ok=True)
    os.makedirs(DST_DETECTED_SKELETONS_FOLDER, exist_ok=True)
    os.makedirs(DST_VIZ_IMGS_FOLDER, exist_ok=True)

    # -- Read images and process
    num_total_images = images_loader.num_images
    for ith_img in range(num_total_images):

        # -- Read image
        try:
            img, str_action_label, img_info = images_loader.read_image()
        except RuntimeError as e:
            print(e)
            continue  # Skip to the next image if an error occurs

        # -- Detect
        humans = skeleton_detector.detect(img)

        # -- Draw
        img_disp = img.copy()
        skeleton_detector.draw(img_disp, humans)
        img_displayer.display(img_disp, wait_key_ms=1)

        # -- Lấy dữ liệu skeleton và lưu vào tệp
        skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
        dict_id2skeleton = multiperson_tracker.track(
            skeletons)  # dict: (int human id) -> (np.array() skeleton)
        skels_to_save = [img_info + skeleton.tolist()
                         for skeleton in dict_id2skeleton.values()]

        # -- Save result

        # Lưu dữ liệu skeleton cho việc huấn luyện
        filename = SKELETON_FILENAME_FORMAT.format(ith_img)
        lib_commons.save_listlist(
            DST_DETECTED_SKELETONS_FOLDER + filename,
            skels_to_save)

        # Lưu hình ảnh được visualized cho mục debug
        filename = IMG_FILENAME_FORMAT.format(ith_img)
        cv2.imwrite(
            DST_VIZ_IMGS_FOLDER + filename,
            img_disp)

        print(f"{ith_img}/{num_total_images} th image "
              f"has {len(skeletons)} people in it")

    print("Program ends")
