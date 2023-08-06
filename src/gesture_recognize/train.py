from pathlib import Path

import joblib
import numpy as np
from sklearn import svm


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """加载数据。"""

    data = np.loadtxt("data\\data.txt", delimiter=',')
    data, label = np.split(data, [-1], axis=1)
    # label = np.eye(6)[label[:, 0].astype(np.int32)]
    return data, np.squeeze(label).astype(np.int32)


if __name__ == "__main__":
    data, label = load_data()
    svm = svm.SVC(kernel='rbf', decision_function_shape="ovo")
    svm.fit(data, np.squeeze(label).astype(np.int32))
    model_dir = Path.cwd() / "model"
    if not model_dir.exists():
        model_dir.mkdir()
    for i in range(60):
        pre = svm.predict(data[i].reshape(1, -1))
        print(pre, label[i])
    joblib.dump(svm, "./model/svm.m")
