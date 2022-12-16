#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "csv_read.h"

#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

typedef struct {
    PyObject_HEAD
    int n_columns;
    struct double_array *data;
}


static int pto_csv_init(pto_csv *self, PyObject *args) {
    int i;
    self->columns = 10;
    for (i = 0; i < self->columns * 5; ++i) {
        self.data.append(i);
    }
    return 0;
}

static PyObject *feed(pto_csv *self, PyObject *args) {
    Py_RETURN_NONE;
}

static PyObject *header(pto_csv *self, PyObject *Py_UNUSED(ignored)) {
    Py_RETURN_NONE;
}

static PyObject *columns(pto_csv *self, PyObject *Py_UNUSED(ignored)) {
    Py_RETURN_NONE;
}

static PyObject *selected(pto_csv *self, PyObject *Py_UNUSED(ignored)) {
    Py_RETURN_NONE;
}

static PyObject *data(pto_csv *self, PyObject *Py_UNUSED(ignored)) {
    npy_intp dims[2];
    PyObject *ndarray;

    dims[0] = self->n_columns;
    dims[1] = self->data->length / self->n_columns;
    ndarray = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, self->data);
    Py_INCREF(ndarray);
    return ndarray;
}

static PyMethodDef pto_csv_methods[] = {
    {"feed", (PyCFunction) feed, METH_VARARGS},
    {"header", (PyCFunction) header, METH_NOARGS},
    {"columns", (PyCFunction) columns, METH_NOARGS},
    {"selected", (PyCFunction) selected, METH_NOARGS},
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
    .tp_name = "clusterline.pto.pto_csv",
    .tp_doc = PyDoc_STR("PTO Process Details CSV"),
    .tp_basicsize = sizeof(pto_csv),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) pto_csv_init,
    .tp_methods = pto_csv_methods,
} ;

PyMODINIT_FUNC PyInit_pto_csv(void) {
    PyObject *m;
    if (PyType_Ready(&PTO_CSV) < 0) return NULL;

    m = PyModule_Create(&module_def);
    if (!m) {
        return NULL;
    }

    Py_INCREF(&PTO_CSV);
    if (PyModule_AddObject(m, "PTO_CSV", (PyObject *) &PTO_CSV) < 0) {
        Py_DECREF(&PTO_CSV);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}

