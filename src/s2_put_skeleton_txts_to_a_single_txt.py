#!/usr/bin/env python
# coding: utf-8

'''
Read multiple skeletons txts and saved them into a single txt.
If an image doesn't have skeleton, discard it.
If an image label is not `CLASSES`, discard it.

Input:
    `skeletons/00001.txt` ~ `skeletons/xxxxx.txt` from `SRC_DETECTED_SKELETONS_FOLDER`.
Output:
    `skeletons_info.txt`. The filepath is `DST_ALL_SKELETONS_TXT`.
'''
'''
Đọc nhiều tập tin txt chứa thông tin về skeletons và lưu chúng vào một tập tin txt duy nhất.
Nếu một hình ảnh không có skeleton, loại bỏ nó.
Nếu nhãn của hình ảnh không nằm trong `CLASSES`, loại bỏ nó.

Input:
    `skeletons/00001.txt` ~ `skeletons/xxxxx.txt` từ `SRC_DETECTED_SKELETONS_FOLDER`.
Output:
    `skeletons_info.txt`. Đường dẫn là `DST_ALL_SKELETONS_TXT`.
'''
import numpy as np
import simplejson
import collections
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    # import utils.lib_feature_proc # This is no needed,
    #   because this script only transfer (part of) the data from many txts to a single txt,
    #   without doing any data analsysis.

    import utils.lib_commons as lib_commons


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings


cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s2_put_skeleton_txts_to_a_single_txt.py"]

CLASSES = np.array(cfg_all["classes"])

SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

SRC_DETECTED_SKELETONS_FOLDER = par(cfg["input"]["detected_skeletons_folder"])
DST_ALL_SKELETONS_TXT = par(cfg["output"]["all_skeletons_txt"])

IDX_PERSON = 0  # Chỉ sử dụng skeleton của người thứ 0 trong mỗi hình ảnh
IDX_ACTION_LABEL = 3  # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]

# -- Helper function


def read_skeletons_from_ith_txt(i):
    ''' 
    Tham số:
        i {int}: số thứ tự của tệp txt chứa skeleton. Chỉ số bắt đầu từ 0.
            Nếu có nhiều người, thì có nhiều dữ liệu skeleton trong tệp này.
    Trả về:
        skeletons_in_ith_txt {list of list}:
            Độ dài của mỗi dữ liệu skeleton được giả định là 41 = 5 thông tin hình ảnh + 36 vị trí xy. 
    '''
    filename = SRC_DETECTED_SKELETONS_FOLDER + \
        SKELETON_FILENAME_FORMAT.format(i)
    skeletons_in_ith_txt = lib_commons.read_listlist(filename)
    return skeletons_in_ith_txt


def get_length_of_one_skeleton_data(filepaths):
    ''' Tìm một tệp txt không rỗng, sau đó lấy độ dài của một dữ liệu skeleton.
    Độ dài dữ liệu nên là 41, trong đó:
    41 = 5 + 36.
        5: [cnt_action, cnt_clip, cnt_image, action_label, filepath]
            Xem utils.lib_io.get_training_imgs_info để biết thêm chi tiết
        36: 18 joints * 2 vị trí xy
    '''
    for i in range(len(filepaths)):
        skeletons = read_skeletons_from_ith_txt(i)
        if len(skeletons):
            skeleton = skeletons[IDX_PERSON]
            data_size = len(skeleton)
            assert(data_size == 41)
            return data_size
    raise RuntimeError(f"No valid txt under: {SRC_DETECTED_SKELETONS_FOLDER}.")


# -- Main
if __name__ == "__main__":
    ''' Đọc nhiều tệp txt chứa thông tin về skeletons và lưu chúng vào một tệp txt duy nhất. '''

    # -- Lấy tên tệp chứa skeleton
    filepaths = lib_commons.get_filenames(SRC_DETECTED_SKELETONS_FOLDER,
                                          use_sort=True, with_folder_path=True)
    num_skeletons = len(filepaths)

    # -- Kiểm tra độ dài dữ liệu của một skeleton
    data_length = get_length_of_one_skeleton_data(filepaths)
    print("Độ dài của một skeleton là {data_length}")

    # -- Đọc skeletons và đẩy vào all_skeletons
    all_skeletons = []
    labels_cnt = collections.defaultdict(int)
    for i in range(num_skeletons):

        # Đọc skeletons từ một tệp txt
        skeletons = read_skeletons_from_ith_txt(i)
        if not skeletons:  # Nếu rỗng, loại bỏ hình ảnh này.
            continue
        skeleton = skeletons[IDX_PERSON]
        label = skeleton[IDX_ACTION_LABEL]
        if label not in CLASSES:  # Nếu nhãn không hợp lệ, loại bỏ hình ảnh này.
            continue
        labels_cnt[label] += 1

        # Push to result
        all_skeletons.append(skeleton)

        # Print
        if i == 1 or i % 100 == 0:
            print("{}/{}".format(i, num_skeletons))

    # -- Save to txt
    with open(DST_ALL_SKELETONS_TXT, 'w') as f:
        simplejson.dump(all_skeletons, f)

    print(f"There are {len(all_skeletons)} skeleton data.")
    print(f"They are saved to {DST_ALL_SKELETONS_TXT}")
    print("Number of each action: ")
    for label in CLASSES:
        print(f"    {label}: {labels_cnt[label]}")
