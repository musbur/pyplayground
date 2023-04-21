#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

static PyObject *init(PyObject *self, PyObject *args) {
    PyObject *sequence;

    if (!PyArg_ParseTuple(args, "O", &sequence)) return NULL;
    if (PySequence_Check(sequence)) {
    } else {
        PyErr_Format(PyExc_ValueError, "Expected Sequence");
        return NULL;
    }
    return sequence;
}

static PyMethodDef methods[] = {
    {"init", init, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
} ;

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name = "tuple_list",
    .m_doc = "Test Extension 1",
    .m_size = -1,
    .m_methods = methods,
} ;

PyMODINIT_FUNC PyInit_tuple_list(void) {
    PyObject *m;

    m = PyModule_Create(&module_def);
    if (!m) {
        return NULL;
    }

    return m;
}

