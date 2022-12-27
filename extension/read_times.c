#ifndef STANDALONE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#else
#define _XOPEN_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define TIME_MARKER "TIME,"
#define SAMPLE_TIME "2022-10-26T17:30:58"
#define TIME_LEN (sizeof SAMPLE_TIME - 1)
#define LINE_LEN  (sizeof SAMPLE_TIME + 1)
#define TIME_FMT "%Y-%m-%dT%H:%M:%S"
#define EOL '\n'

static char *read_line_start(char *line, size_t size, FILE *fh) {
    size_t i = 0;
    int c = 0;
    while (c != EOL && c != EOF) {
        c = fgetc(fh);
        if (i++ < size - 1 && c != EOL) *(line++) = c;
    }
    if (c == EOF) {
        return NULL;
    }
    *line = 0;
    return line;
}

static long ts_from_string(const char *s) {
    char *p;
    struct tm tm;
    long ts;
    p = strptime(s, TIME_FMT, &tm);
    if (p) {
        ts = mktime(&tm);
    } else {
#ifndef STANDALONE
        PyErr_Format(PyExc_ValueError, "Not a valid timestamp: %s", s);
#else
        fprintf(stderr, "Not a valid timestamp: %s\n", s);
#endif
        ts = -1;
    }
    return ts;
}

static int get_dates(FILE *fh, long *ts_first, long *ts_last) {
    char *line = NULL;
    char *first = NULL;
    char *last = NULL;
    int reading = 0;
    if (!(line = malloc(LINE_LEN * 2))) {
#ifndef STANDALONE
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
#else
        fprintf(stderr, "Out of memory\n");
#endif
        return -1;
    } 
    first = line + LINE_LEN;
    *first = 0;
    last = first;
    while (read_line_start(line, LINE_LEN, fh)) {
        if (reading) {
            if (!*first) strcpy(first, line);
            last = line;
        } else if (!strncmp(line, TIME_MARKER, sizeof TIME_MARKER - 1)) {
            reading = 1;
        }
    }
   
    printf("<%s>\n<%s>\n", first, last); 
    *(first + TIME_LEN) = 0;
    *(last + TIME_LEN) = 0;
    *ts_first = ts_from_string(first);
    if (*ts_first < 0) {
        free(line);
        return -2;
    }
    *ts_last = ts_from_string(last);
    if (*ts_first < 0) {
        free(line);
        return -3;
    }
    free(line);
    return 0;
}

#ifndef STANDALONE
static PyObject *read_times(PyObject *self, PyObject *args) {
    FILE *fh;
    const char *fn;
    long t1, t2;
    int r;

    if (!PyArg_ParseTuple(args, "s", &fn)) return NULL; /* exception set by ParseTuple() */
    fh = fopen(fn, "rt");
    if (!fh) return PyErr_SetFromErrnoWithFilename(PyExc_OSError, fn);
    r = get_dates(fh, &t1, &t2);
    fclose(fh);
    if (r < 0) return NULL; /* Exception set by get_dates() */
    return PyTuple_Pack(2, PyLong_FromLong(t1), PyLong_FromLong(t2));
}

static PyMethodDef methods[] = {
    {"read_times", read_times, METH_VARARGS},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "read_times",
    NULL,
    -1,
    methods
} ;

PyMODINIT_FUNC PyInit_read_times(void) {
    return PyModule_Create(&module_def);
}
#else
#define TEST_FILE "ControlJobLog.JSON"
int main(void) {
    FILE *fh;
    long ts_first, ts_last;
    int r;

    fh = fopen(TEST_FILE, "rt");
    if (!fh) {
        fprintf(stderr, "Couldn't open %s\n", TEST_FILE);
        return -1;
    }
    r = get_dates(fh, &ts_first, &ts_last);
    if (r != 0) {
        fprintf(stderr, "get_dates() = %d\n", r);
        fclose(fh);
        return r;
    }
    printf("%ld %ld\n", ts_first, ts_last);
    fclose(fh);
    return 0;
}

#endif

