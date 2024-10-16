#!/usr/bin/env python
# coding: utf-8

''' This script does:
1. Load features and labels from csv files
2. Train the model
3. Save the model to `model/` folder.
'''
''' 
Mô tả:
    1. Load features và labels từ các file csv.
    2. Huấn luyện mô hình.
    3. Lưu mô hình vào thư mục `model/`.
'''
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import classification_report
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if True:  # Bao gồm đường dẫn của dự án vào hệ thống
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_classifier import ClassifierOfflineTrain



def par(path):   # Tiền tố ROOT được thêm vào đường dẫn nếu không là đường dẫn tuyệt đối
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings

# Đọc các cài đặt từ file config.yaml
cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s4_train.py"]
# Danh sách các lớp
CLASSES = np.array(cfg_all["classes"])


SRC_PROCESSED_FEATURES = par(cfg["input"]["processed_features"])
SRC_PROCESSED_FEATURES_LABELS = par(cfg["input"]["processed_features_labels"])

DST_MODEL_PATH = par(cfg["output"]["model_path"])

# -- Functions
# Hàm chia dữ liệu thành tập huấn luyện và tập kiểm tra
def train_test_split(X, Y, ratio_of_test_size):
    ''' Chia dữ liệu huấn luyện theo tỉ lệ '''
    IS_SPLIT_BY_SKLEARN_FUNC = True

    # Use sklearn.train_test_split
    if IS_SPLIT_BY_SKLEARN_FUNC:
        RAND_SEED = 1
        tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
            X, Y, test_size=ratio_of_test_size, random_state=RAND_SEED)

    # Chia dữ liệu huấn luyện và kiểm tra giống nhau
    else:
        tr_X = np.copy(X)
        tr_Y = Y.copy()
        te_X = np.copy(X)
        te_Y = Y.copy()
    return tr_X, te_X, tr_Y, te_Y

# Hàm đánh giá mô hình
def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y):
    ''' Đánh giá độ chính xác và thời gian '''

    # Accuracy
    t0 = time.time()

    tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    print(f"Accuracy on training set is {tr_accu}")

    te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    print(f"Accuracy on testing set is {te_accu}")

    print("Báo cáo độ chính xác:")
    print(classification_report(
        te_Y, te_Y_predict, target_names=classes, output_dict=False))

    # Time cost
    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    print("Thời gian dự đoán mỗi mẫu: "
          "{:.5f} seconds".format(average_time))

    # Vẽ biểu đồ độ chính xác
    axis, cf = lib_plot.plot_confusion_matrix(
        te_Y, te_Y_predict, classes, normalize=False, size=(12, 8))
    plt.show()



# -- Main


def main():

    # -- Đọc dữ liệu đã được xử lý
    print("\nReading csv files of classes, features, and labels ...")
    X = np.loadtxt(SRC_PROCESSED_FEATURES, dtype=float)  # features
    Y = np.loadtxt(SRC_PROCESSED_FEATURES_LABELS, dtype=int)  # labels
    
    # -- Train-test split
    tr_X, te_X, tr_Y, te_Y = train_test_split(
        X, Y, ratio_of_test_size=0.3)
    print("\nSau khi chia thành tập huấn luyện và kiểm tra:")
    print("Kích thước của tập huấn luyện X:    ", tr_X.shape)
    print("Số lượng mẫu trong tập huấn luyện: ", len(tr_Y))
    print("Số lượng mẫu trong tập kiểm tra:   ", len(te_Y))

    # -- Huấn luyện mô hình
    print("\nStart training model ...")
    model = ClassifierOfflineTrain()
    model.train(tr_X, tr_Y)

    # -- Đánh giá mô hình
    print("\nStart evaluating model ...")
    evaluate_model(model, CLASSES, tr_X, tr_Y, te_X, te_Y)

    # -- Save model
    print("\nSave model to " + DST_MODEL_PATH)
    with open(DST_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
