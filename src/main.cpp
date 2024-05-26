#define FOURIER_DESC_SIZE 12
#define PI 3.1415926535

#include <filesystem>
#include <fstream>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;


typedef enum {
    Record,
    Predict
} CaptureMode;


static
void fft_encode(const std::vector<cv::Point>& contours, double* out) {
    size_t byte_size = sizeof(fftw_complex) * contours.size();
    std::complex<double>* _contours = (std::complex<double>*)fftw_malloc(byte_size);
    std::complex<double>* _out = (std::complex<double>*)fftw_malloc(byte_size);
    for (unsigned int i = 0; i < contours.size(); i++) {
        _contours[i] = {(double)contours[i].x, (double)contours[i].y};
    }
    fftw_plan plan = fftw_plan_dft_1d(
        contours.size(), (fftw_complex*)_contours, (fftw_complex*)_out,
        FFTW_FORWARD, FFTW_ESTIMATE
    );
    fftw_execute(plan);
    size_t half = FOURIER_DESC_SIZE / 2;
    double base = std::abs(_out[0]);
    for (unsigned int i = 0; i < half; i++) {
        out[i] = std::abs(_out[i]) / base;
        out[half + i] = std::abs(_out[contours.size() - half + i]) / base;
    }
    // for (unsigned int i = 0; i < FOURIER_DESC_SIZE; i++) {
    //     std::cout << out[i] << std::endl;
    // }
    fftw_destroy_plan(plan);
    fftw_free((void*)_contours);
    fftw_free((void*)_out);
}


static
unsigned int find_max_contour(
    const std::vector<std::vector<cv::Point>>& contours
) {
    unsigned int max_size = 0;
    unsigned int index;
    for (unsigned int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > max_size) {
            max_size = contours[i].size();
            index = i;
        }
    }
    return index;
}


class PyInterpreter {
    private:
        PyObject* pyglobal;

    public:
        PyInterpreter(const char* model) {
            Py_Initialize();
            if (_import_array() < 0) {
                PyErr_Print();
                PyErr_SetString(PyExc_ImportError,
                        "numpy.core.multiarray failed to import");
            }

            char buffer[500];
            sprintf(
                buffer,
                "import joblib\n"
                "model = joblib.load('%s')\n",
                model
            );
            PyRun_SimpleString(buffer);
            pyglobal = PyImport_AddModule("__main__");
        }

        ~PyInterpreter() {
            Py_Finalize();
        }

        int predict(double* fourier_desc, unsigned int length) {
            Py_ssize_t shape[2];
            shape[0] = 1;
            shape[1] = length;
            PyList_New(length);
            auto arr = PyArray_SimpleNewFromData(2, &shape[0], NPY_FLOAT64, (void*)fourier_desc);
            PyObject_SetAttrString(pyglobal, "arr", arr);
            PyRun_SimpleString("res = model.predict(arr)[0]\n");
            auto res = PyObject_GetAttrString(pyglobal, "res");
            if (res == nullptr) {
                PyErr_Print();
                exit(1);
            } else {
                return PyLong_AsLong(res);
            }
        }
};


int main() {
    PyInterpreter py("/home/kewuaa/Projects/gesture_recognize/model/svm.model");
    const char* data_dir = "./data";
    std::ofstream data_file("./data/_data.txt", std::ios::trunc);
    char text[100];
    cv::Size text_size;
    unsigned int label = 0, pre;
    CaptureMode mode = Record;
    if (!fs::exists(data_dir)) {
        fs::create_directories(data_dir);
    }
    cv::VideoCapture cap(0);
    cv::Mat roi, _roi;
    cv::Rect* roi_rect = nullptr;
    cv::Mat frame;
    cv::Mat ycrcb;
    cv::Mat channels[3];
    cv::Mat kernel = cv::getGaussianKernel(5, 5.);
    std::vector<std::vector<cv::Point>> contours;
    double fourier_desc[FOURIER_DESC_SIZE];
    unsigned int index;
    int key;
    while (cap.isOpened()) {
        cap >> frame;
        cv::flip(frame, ycrcb, 1);
        frame = ycrcb;
        unsigned int window_w = frame.cols / 2;
        unsigned int window_h = frame.rows / 2;
        if (roi_rect == nullptr) {
            roi_rect = new cv::Rect(
                frame.cols - window_w, 0,
                window_w, window_h
            );
        }
        _roi = frame(*roi_rect);
        cv::cvtColor(_roi, ycrcb, cv::COLOR_BGR2YCrCb);
        cv::split(ycrcb, channels);
        cv::filter2D(channels[1], ycrcb, CV_8U, kernel);
        cv::threshold(
            ycrcb,
            _roi,
            0,
            255.,
            cv::THRESH_OTSU | cv::THRESH_BINARY
        );
        cv::findContours(
            _roi,
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_NONE
        );
        roi = cv::Mat::zeros(_roi.rows, _roi.cols, CV_8U);
        index = find_max_contour(contours);
        cv::rectangle(frame, *roi_rect, {0, 255, 0});
        cv::drawContours(roi, contours, index, 255);
        fft_encode(contours[index], fourier_desc);;
        switch (mode) {
            case Record: {
                std::sprintf(text, "mode: Record; label: %d\0", label);
                text_size = cv::getTextSize(
                    text,
                    cv::FONT_HERSHEY_COMPLEX,
                    1,
                    1,
                    0
                );
                cv::putText(
                    frame,
                    text,
                    {0, text_size.height},
                    cv::FONT_HERSHEY_COMPLEX,
                    1,
                    {0, 0, 0}
                );
                break;
            }
            case Predict: {
                std::sprintf(text, "mode: Predict\0");
                pre = py.predict(fourier_desc, FOURIER_DESC_SIZE);
                std::sprintf(text + 100, "%d\0", pre);
                text_size = cv::getTextSize(
                    text + 100,
                    cv::FONT_HERSHEY_COMPLEX,
                    1,
                    1,
                    0
                );
                cv::putText(
                    frame,
                    text + 100,
                    { roi_rect->x - text_size.width, text_size.height },
                    cv::FONT_HERSHEY_COMPLEX,
                    1,
                    {0, 0, 255}
                );
                cv::putText(
                    frame,
                    text,
                    {0, text_size.height},
                    cv::FONT_HERSHEY_COMPLEX,
                    1,
                    {0, 0, 0}
                );
                break;
            }
        }
        cv::imshow("camera", frame);
        cv::imshow("roi", roi);
        key = cv::waitKey(1);
        switch (key) {
            // ord of q
            case 113: {
                cap.release();
                break;
            }
            // ord of s
            case 115: {
                if (mode == Predict) break;
                for (unsigned int i = 0; i < FOURIER_DESC_SIZE; i++) {
                    data_file << fourier_desc[i];
                    data_file << ",";
                }
                data_file << label << "\n";
                break;
            }
            // ord of +
            case 43: {
                label += 1;
                break;
            }
            // ord of -
            case 45: {
                label -= 1;
                break;
            }
            // ord of space
            case 32: {
                switch (mode) {
                    case Record: {
                        mode = Predict;
                        break;
                    }
                    case Predict: {
                        mode = Record;
                        break;
                    }
                }
            }
        }
    }
    cv::destroyAllWindows();
    data_file.close();
    if (!fs::exists("./data/data.txt")) {
        fs::rename("./data/_data.txt", "./data/data.txt");
    }
    delete roi_rect;
    return 0;
}
