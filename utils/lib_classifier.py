'''
Script này bao gồm:

1. ClassifierOfflineTrain
    Đây là lớp dành cho việc huấn luyện offline. Dữ liệu đầu vào là các đặc trưng đã được xử lý.
2. Class ClassifierOnlineTest
    Đây là lớp dành cho việc kiểm tra online. Dữ liệu đầu vào là dữ liệu thô từ khung xương.
    Nó sử dụng FeatureGenerator để trích xuất đặc trưng,
    và sau đó sử dụng ClassifierOfflineTrain để nhận dạng hành động.
    Lưu ý, mô hình này chỉ nhận dạng hành động của một người.

    
TODO: Add more comments to this function.
'''

import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

if True:
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    sys.path.append(ROOT)

    from utils.lib_feature_proc import FeatureGenerator


# -- Cài đặt PCA
NUM_FEATURES_FROM_PCA = 50

# -- Classes


class ClassifierOfflineTrain(object):
    ''' Lớp dành cho việc huấn luyện offline.
        Các đặc trưng đầu vào của classifer này đã được 
            xử lý bởi `class FeatureGenerator`.
    '''

    def __init__(self):
        self._init_all_models()
        self.clf = self._choose_model("Neural Net")

    def predict(self, X):
        ''' Dự đoán chỉ số lớp của đặc trưng X '''
        # Sử dụng mô hình đã huấn luyện để dự đoán
        Y_predict = self.clf.predict(self.pca.transform(X))
        return Y_predict

    def predict_and_evaluate(self, te_X, te_Y):
        ''' Kiểm tra mô hình trên tập kiểm tra và tính toán độ chính xác '''
        te_Y_predict = self.predict(te_X)
        N = len(te_Y)
        n = sum(te_Y_predict == te_Y)
        accu = n / N
        return accu, te_Y_predict

    def train(self, X, Y):
        ''' Huấn luyện mô hình. Kết quả được lưu vào self.clf '''
        n_components = min(NUM_FEATURES_FROM_PCA, X.shape[1])
        self.pca = PCA(n_components=n_components, whiten=True)
        self.pca.fit(X)
        # print("Tổng giá trị riêng:", np.sum(self.pca.singular_values_))
        print("Tổng giá trị riêng:", np.sum(self.pca.explained_variance_ratio_))
        X_new = self.pca.transform(X)
        print("Sau khi PCA, X.shape = ", X_new.shape)
        
        # Huấn luyện mô hình với dữ liệu đã qua PCA
        self.clf.fit(X_new, Y)

    def _choose_model(self, name):
        # Chọn mô hình dựa trên tên mô hình
        self.model_name = name
        idx = self.names.index(name)
        return self.classifiers[idx]

    def _init_all_models(self):
        # Khởi tạo các classifier khác nhau
        self.names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                      "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                      "Naive Bayes", "QDA"]
        self.model_name = None
        self.classifiers = [
            KNeighborsClassifier(5),
            SVC(kernel="linear", C=10.0),
            SVC(gamma=0.01, C=1.0, verbose=True),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(
                max_depth=30, n_estimators=100, max_features="auto"),
            MLPClassifier((20, 30, 40)),  # Neural Net
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

    def _predict_proba(self, X):
        ''' Dự đoán xác suất của đặc trưng X thuộc về mỗi lớp Y[i] '''
        Y_probs = self.clf.predict_proba(self.pca.transform(X))
        return Y_probs  # np.array with a length of len(classes)


class ClassifierOnlineTest(object):
    ''' Classifier dành cho dự đoán online.
        Dữ liệu đầu vào của classifier này là dữ liệu khung xương thô, 
        do đó chúng sẽ được xử lý bởi `class FeatureGenerator` trước khi
        được gửi đến mô hình được huấn luyện bởi `class ClassifierOfflineTrain`. 
    '''

    def __init__(self, model_path, action_labels, window_size, human_id=0):

        # -- Settings
        self.human_id = human_id
        # Tải mô hình đã huấn luyện từ file
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        if self.model is None:
            print("my Error: failed to load model")
            assert False
        self.action_labels = action_labels
        self.THRESHOLD_SCORE_FOR_DISP = 0.5

        # -- Lưu trữ thời gian thực
        self.feature_generator = FeatureGenerator(window_size)
        self.reset()

    def reset(self):
        # Đặt lại generator và lịch sử điểm
        self.feature_generator.reset()
        self.scores_hist = deque()
        self.scores = None

    def predict(self, skeleton):
        ''' Dự đoán lớp (string) của khung xương đầu vào '''
        LABEL_UNKNOWN = ""
        is_features_good, features = self.feature_generator.add_cur_skeleton(
            skeleton)

        if is_features_good:
            # Chuyển đổi thành mảng 2 chiều
            features = features.reshape(-1, features.shape[0])

            # Dự đoán xác suất của từng hành động
            curr_scores = self.model._predict_proba(features)[0]
            self.scores = self.smooth_scores(curr_scores)

            if self.scores.max() < self.THRESHOLD_SCORE_FOR_DISP: # Nếu thấp hơn ngưỡng, không đáng tin cậy
                prediced_label = LABEL_UNKNOWN
            else:
                predicted_idx = self.scores.argmax()
                prediced_label = self.action_labels[predicted_idx]
        else:
            prediced_label = LABEL_UNKNOWN
        return prediced_label

    def smooth_scores(self, curr_scores):
        ''' Làm mịn điểm dự đoán hiện tại 
            bằng cách lấy trung bình với các điểm trước đó
        '''
        self.scores_hist.append(curr_scores)
        DEQUE_MAX_SIZE = 2
        if len(self.scores_hist) > DEQUE_MAX_SIZE:
            self.scores_hist.popleft()

        if 1:  # Use sum
            score_sums = np.zeros((len(self.action_labels),))
            for score in self.scores_hist:
                score_sums += score
            score_sums /= len(self.scores_hist)
            print("\nMean score:\n", score_sums)
            return score_sums

        else:  # Use multiply
            score_mul = np.ones((len(self.action_labels),))
            for score in self.scores_hist:
                score_mul *= score
            return score_mul

    def draw_scores_onto_image(self, img_disp):
        if self.scores is None:
            return

        for i in range(-1, len(self.action_labels)):

            FONT_SIZE = 0.7
            TXT_X = 20
            TXT_Y = 150 + i*30
            COLOR_INTENSITY = 255

            if i == -1:
                s = "P{}:".format(self.human_id)
            else:
                label = self.action_labels[i]
                s = "{:<5}: {:.2f}".format(label, self.scores[i])
                COLOR_INTENSITY *= (0.0 + 1.0 * self.scores[i])**0.5

            cv2.putText(img_disp, text=s, org=(TXT_X, TXT_Y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SIZE,
                        color=(0, 0, int(COLOR_INTENSITY)), thickness=2)
