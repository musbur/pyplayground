#ifndef __PTO_CSV_H__
#define __PTO_CSV_H__

#include "csv_read.h"

enum {
    PCSV_OK,            // no error
    PCSV_ERR_DATA,      // Unexpected data in file
    PCSV_ERR_MALLOC,    // malloc() fail
    PCSV_ERR_HEADER,    // error in texts before data
    PCSV_ERR_STRUCTURE, // inconsistent line lengths
    PCSV_ERR_NUMBER,    // Float conversion failure
    PCSV_ERR_INTERNAL   // internal erroe (= bug)
} ;

extern int pcsv_errno;

typedef void (free_func_t) (void *p);

struct ll {
    struct ll *next;
    void *data;
    free_func_t *free_func;
} ;

struct list {
    struct ll *root;
    struct ll *head;
    int length;
} ;

struct double_array {
    double *value;
    double *pos;
    int length;
    int allocated;
} ;

struct pto_context {
    /* Public interface. Read only. Valid only AFTER pcsv_feed()
     * is exhausted without errors. */
    struct list *header_keys;    // For the general header (Wafer ID, ..)
    struct list *header_values;
    struct list *colnames;       // All colnames found in file
    char *column_mask;           // non-zero for each selected col
    struct double_array *data;

    /* private */
    char **colname_patterns; /* NULL-terminated pattern on which 
                                colnames must END */
    struct list *current_list;
    int error;
    int status;
    int current_line;
    int current_col;
    struct csv_context *csv;
} ;


struct pto_context *pcsv_new(void);

int pcsv_feed(struct pto_context *ctx, const char *buffer, int buflen);

void pcsv_free(struct pto_context *ctx);

#endif
