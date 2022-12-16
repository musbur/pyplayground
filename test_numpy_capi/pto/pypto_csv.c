#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "csv_read.h"

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
    int current_line;
    int current_col;
    struct ll *header_keys;
    struct ll *header_values;
    struct ll *colnames;
    struct ll *ll_p;
    int n_columns;
    struct double_array *data;
    char *column_mask;
    struct csv_context *csv;
    PyObject *colname_patterns;
} pto_csv;

static PyObject *create_ndarray(PyObject *self, PyObject *args) {
    /* This needs to go into a type class in order to be able to free the
     * array after use */
    int size;
    int i;
    npy_intp dims[1];
    double *array;
    PyObject *ndarray;

    if (!PyArg_ParseTuple(args, "i", &size)) return NULL;

    array = malloc(size * sizeof *array);
    for (i = 0; i < size; ++i) {
        array[i] = (double) i / (double) size;
    }
    dims[0] = size;
    ndarray = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, array);
    Py_INCREF(ndarray);
    return ndarray;
}

int get_text(const char *piece, off_t pos, void *user_data) {
    pto_csv *ctx = user_data;
    (void) pos;
    if (ctx->ll_p) ctx->ll_p = ll_append(ctx->ll_p, piece);
    return 0;
}

int get_double(const char *piece, off_t pos, void *user_data) {
    pto_csv *ctx = user_data;
    double v;
    char *e;
    (void) pos;
    if (ctx->current_col >= ctx->n_columns) {
        return 0; // silently ignore excess value
    }
    if (ctx->column_mask[ctx->current_col]) {
        if (ctx->current_col == 0) {
            static struct tm tm;
            char *p;
            p = strptime(piece, "%Y-%m-%dT%H:%M:%S", &tm);
            if (p) {
                v = mktime(&tm);
            } else {
                fprintf(stderr, "Not a time: %s\n", piece);
                v = -1;
            }
        } else {
            e = (char *) piece;
            v = strtod(piece, &e);
            if (*e != 0) {
                fprintf(stderr, "Not a float: %s\n", piece);
                v = -1;
            }
        }
        double_a_append(ctx->data, v);
    }
    ++ctx->current_col;
    return 0;
}

int get_line(int line_no, int n_pieces, void *user_data) {
    pto_csv *ctx = user_data;
    int i;
//    fprintf(stderr, "Line %d\n", (int) line_no);
    switch (line_no) {
        case 0:
        case 1:
        case 2:
        case 3:
            break;
        case 4:
            ctx->ll_p = ctx->header_keys;
            break;
        case 5:
            ctx->ll_p = ctx->header_values;
            break;
        case 6:
            ctx->ll_p = ctx->colnames;
            break;
        case 7:
            ctx->n_columns = n_pieces;
            ctx->column_mask = calloc(ctx->n_columns, 1);
            ctx->data = double_a_new(300);
            for (i = 0; i < ctx->n_columns; ++i) {
                ctx->column_mask[i] = 1;
            }
            ctx->csv->func_piece = get_double;
            fprintf(stderr, "Got %d columns\n", ctx->n_columns);
            break;
        default:
            if (n_pieces != ctx->n_columns) {
                fprintf(stderr, "Expected %d values in line %d, got %d\n",
                        (int) ctx->n_columns,
                        (int) line_no,
                        (int) n_pieces);
            }
            break;
    }
    ctx->current_col = 0;
    return 0;
}

static int pto_csv_init(pto_csv *self, PyObject *args) {
    PyObject *colnames;
    int i;
    if (!PyArg_ParseTuple(args, "O", &colnames)) return -1;
    Py_INCREF(colnames);
    if (!(self->header_keys = ll_new())) return -1;
    if (!(self->header_values = ll_new())) return -1;
    if (!(self->colnames = ll_new())) return -1;
    self->ll_p = NULL;
    self->current_line = 1;
    self->current_col = -1;
    self->n_columns = -1;
    self->column_mask = NULL;
    self->colname_patterns = colnames;
    self->csv = csv_new(',', get_text, get_line, self);
    return 0;
}

static PyObject *feed(pto_csv *self, PyObject *args) {
    Py_sint buflen;
    char *buffer;
    int r;
    if (!PyArg_ParseTuple(args, "y#", &buffer, &buflen)) return NULL;
    r = csv_feed(self->csv, buffer, buflen);
    if (r != 0) {
        return PyErr_Format(PyExc_RuntimeError, "csv_feed() = %d", r);
    }
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

