#ifndef __PTO_CSV_H__
#define __PTO_CSV_H__

#include "csv_read.h"

typedef struct  {
    /* publich interface, please read-only */
    struct ll *header_keys;
    struct ll *header_values;
    struct ll *colnames;
    char **colname_patterns; /* NULL-terminated pattern on which 
                                colnames must END */
    struct double_array *data;
    int error;
    /* private stuff */
    int status;
    int n_columns;
    char *column_mask;
    int current_line;
    int current_col;
    struct ll *ll_p;
    struct csv_context *csv;
} pto_context ;

pto_context *pcsv_new(void);

int pcsv_feed(pto_context *ctx, const char *buffer, int buflen);

void pcsv_free(pto_context *ctx);

#endif
