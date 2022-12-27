#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pto_csv.h"

#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#define TIME_MARKER "TIME,"
#define SAMPLE_TIME "2022-10-26T17:30:58"
#define TIME_LEN (sizeof SAMPLE_TIME - 1)
#define LINE_LEN  (sizeof SAMPLE_TIME + 1)
#define TIME_FMT "%Y-%m-%dT%H:%M:%S"
#define EOL '\n'

typedef struct {
    PyObject_HEAD
    struct pto_context *context;
    PyObject *colname_patterns;
    char **colpat;
} pto_csv;

static int pto_csv_init(pto_csv *self, PyObject *args) {
    int npat, i;
    PyObject *arg;

    self->context = NULL;
    self->colname_patterns = NULL;
    self->colpat = NULL;
    self->context = pcsv_new();
    if (!self->context) goto MALLOC_ERROR;

    if (!PyArg_ParseTuple(args, "O", &self->colname_patterns)) return -1;
    if (PySequence_Check(self->colname_patterns)) {
        npat = PySequence_Size(self->colname_patterns);
        if (!(self->colpat = malloc((npat + 1) * sizeof *self->colpat))) {
            goto MALLOC_ERROR;
        }
        for (i = 0; i < npat; ++i) {
            PyObject *str;
            str = PySequence_GetItem(self->colname_patterns, i);
            self->colpat[i] = PyUnicode_DATA(str);
        }
        self->colpat[npat] = NULL;
    } else if (self->colname_patterns != Py_None) {
        PyErr_Format(PyExc_ValueError, "Expected tuple or None");
        return -1;
    }

    self->context->colname_patterns = self->colpat;

    goto CONTINUE;
MALLOC_ERROR:
    pcsv_free(self->context);
    return -1;
CONTINUE:
    return 0;
}

static PyObject *feed(pto_csv *self, PyObject *args) {
    Py_ssize_t buflen;
    const char *buffer;
    char *x;
    int r;
    if (!PyArg_ParseTuple(args, "y#", &buffer, &buflen)) return NULL;
    x = malloc(buflen + 1);
    strncpy(x, buffer, buflen);
    x[buflen] = 0;
    r = pcsv_feed(self->context, buffer, buflen);
    if (r != 0) {
        return PyErr_Format(PyExc_RuntimeError, "csv_feed(): %s",
               pcsv_errmsg[pcsv_errno]);
    }
    Py_RETURN_NONE;
}

static PyObject *header(pto_csv *self, PyObject *Py_UNUSED(ignored)) {
    Py_RETURN_NONE;
}

static PyObject *columns(pto_csv *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *clist;
    struct ll *lp;
    if (!(clist = PyList_New(0))) return NULL;

    for (lp = self->context->colnames->root; lp; lp = lp->next) {
        PyList_Append(clist, PyUnicode_FromString(lp->data));
    }
    Py_INCREF(clist);
    return clist;
}

static PyObject *selected(pto_csv *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *clist;
    int i;
    struct ll *lp;

    if (!(clist = PyList_New(0))) return NULL;

    i = 0;
    for (lp = self->context->colnames->root; lp; lp = lp->next) {
        if (self->context->column_mask[i]) {
            PyList_Append(clist, PyUnicode_FromString(lp->data));
        }
        ++i;
    }
    Py_INCREF(clist);
    return clist;
}

static void dealloc(pto_csv *self) {
    pcsv_free(self->context);
    free(self->colpat);
    if (self->colname_patterns) {
        Py_DECREF(self->colname_patterns);
    }
    Py_TYPE(self)->tp_free(self);
}

static PyObject *data(pto_csv *self, PyObject *Py_UNUSED(ignored)) {
    npy_intp dims[2];
    PyObject *ndarray;
    int i, n_columns;

    for (n_columns = 0, i = 0; i < self->context->colnames->length; ++i) {
        if (self->context->column_mask[i]) ++n_columns;
    }
    fprintf(stderr, "%d %d\n", self->context->data->length, n_columns);
    Py_INCREF(Py_None);

    dims[0] = self->context->data->length / n_columns;
    dims[1] = n_columns;
    ndarray = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE,
                                        self->context->data->value);
    Py_INCREF(ndarray);
    return ndarray;
}

static PyMethodDef pto_csv_methods[] = {
    {"feed",     (PyCFunction) feed,     METH_VARARGS},
    {"header",   (PyCFunction) header,   METH_NOARGS},
    {"columns",  (PyCFunction) columns,  METH_NOARGS},
    {"selected", (PyCFunction) selected, METH_NOARGS},
    {"data",     (PyCFunction) data,     METH_NOARGS},
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
    .tp_dealloc = (destructor) dealloc,
    .tp_methods = pto_csv_methods,
} ;

PyMODINIT_FUNC PyInit_pto_csv(void) {
    PyObject *m;
    if (PyType_Ready(&PTO_CSV) < 0) return NULL;

    m = PyModule_Create(&module_def);
    if (!m) {
        return NULL;
    }

    import_array();

    Py_INCREF(&PTO_CSV);
    if (PyModule_AddObject(m, "PTO_CSV", (PyObject *) &PTO_CSV) < 0) {
        Py_DECREF(&PTO_CSV);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}

