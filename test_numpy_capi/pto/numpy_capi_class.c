#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

typedef struct {
    PyObject_HEAD
} pto_csv;

static int init(pto_csv *self, PyObject *args) {
    return 0;
}

static PyObject *data(pto_csv *self, PyObject *Py_UNUSED(ignored)) {
    npy_intp dims[2];
    PyObject *ndarray;
    int i;

    double *a;
    a = malloc(100 * sizeof *a);
    for (i = 0; i < 100; ++i) a[i] = i;
        
    dims[0] = 10;
    dims[1] = 10;
    ndarray = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, a);
    Py_INCREF(ndarray);
    return ndarray;
}

static PyMethodDef pto_csv_methods[] = {
    {"data", (PyCFunction) data, METH_NOARGS},
    {NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pto_csv",
    .m_doc = "Defines a class PTO_CSV for reading PTO process logs",
    .m_size = -1,
} ;

static PyTypeObject PTO_CSV = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pto_csv",
    .tp_doc = PyDoc_STR("PTO Process Details CSV"),
    .tp_basicsize = sizeof(pto_csv),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) init,
    .tp_methods = pto_csv_methods,
} ;

PyMODINIT_FUNC PyInit_numpy_capi_class(void) {
    PyObject *m;
    m = PyModule_Create(&module_def);

    if (!m) {
        return NULL;
    }

    import_array();

    if (PyType_Ready(&PTO_CSV) < 0) return NULL;
    Py_INCREF(&PTO_CSV);
    if (PyModule_AddObject(m, "PTO_CSV", (PyObject *) &PTO_CSV) < 0) {
        Py_DECREF(&PTO_CSV);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}

