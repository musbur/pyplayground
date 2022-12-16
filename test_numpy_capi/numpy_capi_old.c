#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>

#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

static PyObject *create_ndarray(PyObject *self, PyObject *args) {
    /* This needs to go into a type class in order to be able to free the
     * array after use */
    int dim1;
    int dim2;
    int i, j;
    npy_intp dims[2];
    double *array;
    PyObject *ndarray;

    if (!PyArg_ParseTuple(args, "ii", dim1, dim2)) return NULL;

    array = malloc(dim1 * dim2 * sizeof *array);
    for (i = 0; i < dim1; ++i) {
        for (j = 0; j < dim2; ++j) {
            array[dim2 * i + j] = 10 * j + (double) i / (double) dim1;
        }
    }
    dims[0] = dim1;
    dims[1] = dim2;
    ndarray = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, array);
    Py_INCREF(ndarray);
    return ndarray;
}

static PyObject *plusone(PyObject *self, PyObject *args) {
    int size;
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "i", &size)) return NULL;
    obj = PyLong_FromLong(size + 1);
    Py_INCREF(obj);
    fprintf(stderr, "%d %p\n", size, obj);
    return obj;
}

static PyObject *hello(PyObject *self, PyObject *args) {
    const char *name;
    if (!PyArg_ParseTuple(args, "s", &name)) return NULL;
    return PyUnicode_FromFormat("Hello %s!", name);
}

static PyMethodDef module_methods[] = {
    {"hello", hello, METH_VARARGS,
        "Sanity check"},
    {"plusone", plusone, METH_VARARGS,
        "Another test"},
    {"create_ndarray", create_ndarray, METH_VARARGS,
        "Create a NumPy ndarray"},
    {NULL, NULL, 0, NULL},
} ;

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "numpy_capi",
    "Tesing Numpy's C API",
    -1,
    module_methods
} ;

PyMODINIT_FUNC PyInit_numpy_capi (void) {
    PyObject *m;

    import_array();
    m = PyModule_Create(&module);
    return m;
}
