#!/usr/bin/env python
# coding: utf-8

'''
Kiểm tra nhận dạng hành động trên
(1) một video, (2) một thư mục chứa ảnh, (3) hoặc máy ảnh web.

Nhập vào:
    model: model/trained_classifier.pickle

Xuất ra:
    video kết quả:    output/${video_name}/video.avi
    kết quả xương sống: output/${video_name}/skeleton_res/XXXXX.txt
    hiển thị bằng cv2.imshow() trong img_displayer
'''

'''
Ví dụ về cách sử dụng:

(1) Kiểm tra trên tệp video:
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type video \
    --data_path data_test/exercise.avi \
    --output_folder output
    
(2) Kiểm tra trên một thư mục chứa ảnh:
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type folder \
    --data_path data_test/apple/ \
    --output_folder output

(3) Kiểm tra trên máy ảnh web:
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type webcam \
    --data_path 0 \
    --output_folder output
    
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import cv2
import argparse
if True:  # Bao gồm đường dẫn dự án
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_images_io as lib_images_io
    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_tracker import Tracker
    from utils.lib_classifier import ClassifierOnlineTest
    from utils.lib_classifier import *  # Nhập tất cả các thư viện liên quan đến sklearn


def par(path):  # Tiền tố ROOT vào đường dẫn nếu nó không phải là đường dẫn tuyệt đối
    return ROOT + path if (path and path[0] != "/") else path


# -- Nhập dữ liệu dòng lệnh


def get_command_line_arguments():

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Kiểm tra nhận dạng hành động trên \n"
            "(1) một video, (2) một thư mục chứa ảnh, (3) hoặc máy ảnh web.")
        parser.add_argument("-m", "--model_path", required=False,
                            default='model/trained_classifier.pickle')
        parser.add_argument("-t", "--data_type", required=False, default='webcam',
                            choices=["video", "folder", "webcam"])
        parser.add_argument("-p", "--data_path", required=False, default="",
                            help="đường dẫn đến tệp video hoặc thư mục chứa ảnh hoặc máy ảnh web. \n"
                            "Đối với video và thư mục, đường dẫn nên là "
                            "tuyệt đối hoặc tương đối với gốc dự án này. "
                            "Đối với máy ảnh web, nhập chỉ số hoặc tên thiết bị. ")
        parser.add_argument("-o", "--output_folder", required=False, default='output/',
                            help="Thư mục để lưu kết quả.")

        args = parser.parse_args()
        return args
    args = parse_args()
    if args.data_type != "webcam" and args.data_path and args.data_path[0] != "/":
        # Nếu đường dẫn không phải là tuyệt đối, thì nó tương đối với ROOT.
        args.data_path = ROOT + args.data_path
    return args


def get_dst_folder_name(src_data_type, src_data_path):
    ''' Tính toán tên thư mục đầu ra dựa trên data_type và data_path.
        Kết quả cuối cùng của kịch bản này sẽ trông như sau:
            DST_FOLDER/folder_name/vidoe.avi
            DST_FOLDER/folder_name/skeletons/XXXXX.txt
    '''

    assert(src_data_type in ["video", "folder", "webcam"])

    if src_data_type == "video":  # /root/data/video.avi --> video
        folder_name = os.path.basename(src_data_path).split(".")[-2]

    elif src_data_type == "folder":  # /root/data/video/ --> video
        folder_name = src_data_path.rstrip("/").split("/")[-1]

    elif src_data_type == "webcam":
        # tháng-ngày-giờ-phút-giây, ví dụ: 02-26-15-51-12
        folder_name = lib_commons.get_time_string()

    return folder_name


args = get_command_line_arguments()

SRC_DATA_TYPE = args.data_type
SRC_DATA_PATH = args.data_path
SRC_MODEL_PATH = args.model_path

DST_FOLDER_NAME = get_dst_folder_name(SRC_DATA_TYPE, SRC_DATA_PATH)

# -- Thiết lập

cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s5_test.py"]

CLASSES = np.array(cfg_all["classes"])
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

# Nhận dạng hành động: số khung hình được sử dụng để trích xuất đặc trưng.
WINDOW_SIZE = int(cfg_all["features"]["window_size"])

# Thư mục đầu ra
DST_FOLDER = args.output_folder + "/" + DST_FOLDER_NAME + "/"
DST_SKELETON_FOLDER_NAME = cfg["output"]["skeleton_folder_name"]
DST_VIDEO_NAME = cfg["output"]["video_name"]
# tốc độ khung hình của video.avi đầu ra
#DST_VIDEO_FPS = 10.0
DST_VIDEO_FPS = float(cfg["output"]["video_fps"])


# Cài đặt Video

# Nếu data_type là webcam, đặt tốc độ khung hình tối đa.
SRC_WEBCAM_MAX_FPS = float(cfg["settings"]["source"]
                           ["webcam_max_framerate"])

# Nếu data_type là video, đặt khoảng lấy mẫu.
# Ví dụ: nếu là 3, thì video sẽ được đọc nhanh hơn 3 lần.
SRC_VIDEO_SAMPLE_INTERVAL = int(cfg["settings"]["source"]
                                ["video_sample_interval"])

# Cài đặt Openpose
OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"]
OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"]

# Cài đặt Hiển thị
img_disp_desired_rows = int(cfg["settings"]["display"]["desired_rows"])


# -- Hàm


def select_images_loader(src_data_type, src_data_path):
    if src_data_type == "video":
        images_loader = lib_images_io.ReadFromVideo(
            src_data_path,
            sample_interval=SRC_VIDEO_SAMPLE_INTERVAL)

    elif src_data_type == "folder":
        images_loader = lib_images_io.ReadFromFolder(
            folder_path=src_data_path)

    elif src_data_type == "webcam":
        if src_data_path == "":
            webcam_idx = 0
        elif src_data_path.isdigit():
            webcam_idx = int(src_data_path)
        else:
            webcam_idx = src_data_path
        images_loader = lib_images_io.ReadFromWebcam(
            SRC_WEBCAM_MAX_FPS, webcam_idx)
    return images_loader


class MultiPersonClassifier(object):
    ''' Đây là một bọc xung quanh ClassifierOnlineTest
        để nhận dạng hành động của nhiều người.
    '''

    def __init__(self, model_path, classes):

        self.dict_id2clf = {}  # id của người -> bộ phân loại của người này

        # Xác định một hàm để tạo bộ phân loại cho những người mới.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, WINDOW_SIZE, human_id)

    def classify(self, dict_id2skeleton):
        ''' Phân loại loại hành động của mỗi cấu trúc xương trong dict_id2skeleton '''

        # Xóa những người không trong tầm nhìn
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Dự đoán hành động của từng người
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.dict_id2clf:  # thêm người mới này
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton)  # dự đoán nhãn
            # print("\n\nDự đoán nhãn cho người{}".format(id))
            # print("  cấu trúc xương: {}".format(skeleton))
            # print("  nhãn: {}".format(id2label[id]))

        return id2label

    def get_classifier(self, id):
        ''' Lấy bộ phân loại dựa trên id của người.
        Đối số:
            id {int hoặc "min"}
        '''
        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


def remove_skeletons_with_few_joints(skeletons):
    ''' Loại bỏ cấu trúc xương không tốt trước khi gửi đến bộ theo dõi '''
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # NẾU CÁC KHỚP ĐANG THIẾU, HÃY THỬ THAY ĐỔI CÁC GIÁ TRỊ NÀY:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            # thêm cấu trúc xương này chỉ khi tất cả các yêu cầu được thỏa mãn
            good_skeletons.append(skeleton)
    return good_skeletons


def draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                    skeleton_detector, multiperson_classifier):
    ''' Vẽ cấu trúc xương, nhãn và điểm dự đoán lên ảnh để hiển thị '''

    # Thay đổi kích thước để phù hợp với hiển thị
    r, c = img_disp.shape[0:2]
    desired_cols = int(1.0 * c * (img_disp_desired_rows / r))
    img_disp = cv2.resize(img_disp,
                          dsize=(desired_cols, img_disp_desired_rows))

    # Vẽ cấu trúc xương của tất cả mọi người
    skeleton_detector.draw(img_disp, humans)

    # Vẽ hộp giới hạn và nhãn của mỗi người
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            # tỷ lệ dữ liệu y trở lại ban đầu
            skeleton[1::2] = skeleton[1::2] / scale_h
            # print("Vẽ cấu trúc xương: ", dict_id2skeleton[id], "với nhãn:", label, ".")
            lib_plot.draw_action_result(img_disp, id, skeleton, label)

    # Thêm không gian trắng bên trái để hiển thị điểm dự đoán của mỗi lớp
    img_disp = lib_plot.add_white_region_to_left_of_image(img_disp)

    cv2.putText(img_disp, "Frame:" + str(ith_img),
                (20, 20), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_PLAIN,
                color=(0, 0, 0), thickness=2)

    # Vẽ điểm dự đoán cho chỉ 1 người
    if len(dict_id2skeleton):
        classifier_of_a_person = multiperson_classifier.get_classifier(
            id='min')
        classifier_of_a_person.draw_scores_onto_image(img_disp)
    return img_disp


def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton):
    '''
    Trong mỗi ảnh, cho mỗi cấu trúc xương, lưu:
        human_id, nhãn và vị trí cấu trúc xương của độ dài 18*2.
    Vì vậy, tổng chiều dài mỗi hàng là 2+36=38
    '''
    skels_to_save = []
    for human_id in dict_id2skeleton.keys():
        label = dict_id2label[human_id]
        skeleton = dict_id2skeleton[human_id]
        skels_to_save.append([[human_id, label] + skeleton.tolist()])
    return skels_to_save


# -- Chính
if __name__ == "__main__":

    # -- Bộ phát hiện, bộ theo dõi, bộ phân loại

    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)

    multiperson_tracker = Tracker()

    multiperson_classifier = MultiPersonClassifier(SRC_MODEL_PATH, CLASSES)

    # -- Người đọc hình ảnh và hiển thị
    images_loader = select_images_loader(SRC_DATA_TYPE, SRC_DATA_PATH)
    img_displayer = lib_images_io.ImageDisplayer()

    # -- Khởi tạo đầu ra

    # thư mục đầu ra
    os.makedirs(DST_FOLDER, exist_ok=True)
    os.makedirs(DST_FOLDER + DST_SKELETON_FOLDER_NAME, exist_ok=True)

    # bộ ghi video
    video_writer = lib_images_io.VideoWriter(
        DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)

    # -- Đọc ảnh và xử lý
    try:
        ith_img = -1
        while images_loader.has_image():

            # -- Đọc ảnh
            img = images_loader.read_image()
            ith_img += 1
            img_disp = img.copy()
            print(f"\nĐang xử lý ảnh thứ {ith_img} ...")

            # -- Phát hiện cấu trúc xương
            humans = skeleton_detector.detect(img)
            skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
            skeletons = remove_skeletons_with_few_joints(skeletons)

            # -- Theo dõi người
            dict_id2skeleton = multiperson_tracker.track(
                skeletons)  # id người -> cấu trúc xương np.array()

            # -- Nhận dạng hành động của mỗi người
            if len(dict_id2skeleton):
                dict_id2label = multiperson_classifier.classify(
                    dict_id2skeleton)

            # -- Vẽ
            img_disp = draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                                       skeleton_detector, multiperson_classifier)

            # In nhãn của một người
            if len(dict_id2skeleton):
                min_id = min(dict_id2skeleton.keys())
                print("Nhãn dự đoán là:", dict_id2label[min_id])

            # -- Hiển thị ảnh và ghi vào video.avi
            img_displayer.display(img_disp, wait_key_ms=1)
            video_writer.write(img_disp)

            # -- Lấy dữ liệu cấu trúc xương và lưu vào tệp
            skels_to_save = get_the_skeleton_data_to_save_to_disk(
                dict_id2skeleton)
            lib_commons.save_listlist(
                DST_FOLDER + DST_SKELETON_FOLDER_NAME +
                SKELETON_FILENAME_FORMAT.format(ith_img),
                skels_to_save)
            # Kiểm tra phím nhấn Esc để ngắt chương trình
            import keyboard
            if keyboard.is_pressed('esc'):
                break

            # Kiểm tra phím nhấn Q để tạm dừng chương trình
            if keyboard.is_pressed('q'):
                while True:
                    if keyboard.is_pressed('q'):
                        break
    finally:
        video_writer.stop()
        print("Chương trình kết thúc")
