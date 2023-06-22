# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language=c
from sklearn import svm
import joblib
cimport numpy as cnp
cnp.import_array()
model: svm.SVC = joblib.load("./model/svm.m")


cdef public int predict(double* fourier_desc, unsigned int length):
    cdef Py_ssize_t shape[2]
    shape[0] = 1
    shape[1] = length
    cdef cnp.ndarray _fourier_desc = <cnp.ndarray>cnp.PyArray_SimpleNewFromData(
        2, shape, cnp.NPY_FLOAT64, <void*>fourier_desc
    )
    return model.predict(_fourier_desc)
