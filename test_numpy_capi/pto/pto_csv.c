#define _XOPEN_SOURCE

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "pto_csv.h"

#define TIME_COLUMN "TIME"
#define ALLOC_INCREMENT 1000

#define STATUS_START 0
#define STATUS_FEED 2
#define STATUS_DATA 3

/*
 *
 *            Some helper data structures
 *
 */

const char *pcsv_errmsg[] = {
    "No error",
    "Unexpected data in file",
    "malloc() fail",
    "Error in header",
    "Inconsistent line lengths",
    "Float conversion failure",
    "Internal error",
} ;

int pcsv_errno = PCSV_OK;

typedef int (list_cb)(int i, void *p, void *user_data);

static struct list *list_new(void) {
    struct list *new;
    static struct list zero;
    if (!(new = malloc(sizeof *new))) {
        pcsv_errno = PCSV_ERR_MALLOC;
        return NULL;
    }
    *new = zero;
    return new;
}

static int list_append(struct list *l, void *data, free_func_t *ff) {
    struct ll *el;
    if (!(el = malloc(sizeof *el))) {
        pcsv_errno = PCSV_ERR_MALLOC;
        return -1;
    }
    el->data = data;
    el->free_func = ff;
    el->next = NULL;
    if (l->head) {
        l->head->next = el;
        l->head = l->head->next;
    } else {
        l->root = el;
        l->head = l->root;
    }
    return ++l->length;
}

int list_append_str(struct list *l, const char *str) {
    char *s;
    if (!(s = malloc(strlen(str) + 1))) {
        pcsv_errno = PCSV_ERR_MALLOC;
        return -1;
    }

    return list_append(l, strcpy(s, str), free);
}

void **list_to_array(struct list *l) {
    void **arr;
    int i;
    struct ll *p;

    if (!(arr = malloc((l->length + 1) * sizeof *arr))) {
        pcsv_errno = PCSV_ERR_MALLOC;
        return NULL;
    }
    for (i = 0, p = l->root; i < l->length; ++i) {
        if (!p) {
            free(arr);
            pcsv_errno = PCSV_ERR_INTERNAL;
            return NULL;
        }
        arr[i] = p->data;
    }
    return arr;
}

void list_free(struct list *l) {
    struct ll *lp, *next;
    if (!l) return;
    lp = l->root;
    while (lp) {
        if (lp->free_func) lp->free_func(lp->data);
        next = lp->next;
        free(lp);
        lp = next;
    }
    free(l);
}

void list_print(FILE *fp, struct list *l) {
    int i;
    struct ll *p;
    if (!l) return;
    for (p = l->root, i = 0; p && i < l->length; ++i) {
        fprintf(fp, "%2d: %s\n", i, (char *) p->data);
        p = p->next;
    }
}

struct double_array *double_a_new() {
    struct double_array *arr;
    static struct double_array zero;
    if (!(arr = malloc(sizeof *arr))) {
        pcsv_errno = PCSV_ERR_MALLOC;
        return NULL;
    }
    *arr = zero;
    return arr;
}

int double_a_append(struct double_array *arr, double v) {
    if (arr->length == arr->allocated) {
        double *t;
        arr->allocated += ALLOC_INCREMENT;
        t = realloc(arr->value, arr->allocated * sizeof *t);
        if (!t) {
            pcsv_errno = PCSV_ERR_MALLOC;
            return -1;
        }
        arr->value = t;
        arr->pos = arr->value + arr->length;
    }
    *(arr->pos++) = v;
    ++arr->length;
    return 0;
}

void double_a_free(struct double_array *arr) {
    if (!arr) return;
    free(arr->value);
    free(arr);
}

/*
 *
 *            get_...(): Callbacks doing the actual work
 *
 */

static int get_text(const char *piece, long pos, void *user_data) {
    struct pto_context *ctx = user_data;
    (void) pos;
    if (ctx->current_list) list_append_str(ctx->current_list, piece);
    return 0;
}

static int get_double(const char *piece, long pos, void *user_data) {
    struct pto_context *ctx = user_data;
    double v;
    char *e;
    (void) pos;
    if (ctx->current_col >= ctx->colnames->length) {
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
                pcsv_errno = PCSV_ERR_DATA;
                return -1;
            }
        } else {
            e = (char *) piece;
            v = strtod(piece, &e);
            if (*e != 0) {
                pcsv_errno = PCSV_ERR_NUMBER;
                return -1;
            }
        }
        double_a_append(ctx->data, v);
    }
    ++ctx->current_col;
    return 0;
}

static int endswith(const char *str, const char *pat) {
    // Returns nonzero if pat matches the end of str
    int diff = strlen(str) - strlen(pat);
    if (diff < 0) return 0;
    return (strcmp(pat, str + diff) == 0);
}

static char *make_colmask(struct list *colnames, char **wanted) {
    char *mask;
    char **wp;
    int i;
    struct ll *lp;

    if (!(mask = malloc(colnames->length))) {
        pcsv_errno = PCSV_ERR_MALLOC;
        return NULL;
    }
    for (lp = colnames->root, i = 0; i < colnames->length; ++i) {
        mask[i] = 0;
        if (wanted) {
            for (wp = wanted; *wp; ++wp) {
                if (endswith(lp->data, *wp)) {
                    mask[i] = 1;
                    break;
                }
            }
        } else {
            mask[i] = 1;
        }
        lp = lp->next;
        if (!lp) break;
    }
    return mask;
}

static int get_line(int line_no, int n_pieces, void *user_data) {
    struct pto_context *ctx = user_data;
    switch (line_no) {
        case 0:
        case 1:
        case 2:
        case 3:
            fprintf(stderr, "READ LINE NO %d\n", line_no);
            break;
        case 4:
            ctx->current_list = ctx->header_keys;
            break;
        case 5:
            ctx->current_list = ctx->header_values;
            break;
        case 6:
            ctx->current_list = ctx->colnames;
            break;
        case 7:
            ctx->column_mask = make_colmask(ctx->colnames,
                                            ctx->colname_patterns);
            ctx->data = double_a_new();
            ctx->csv->func_piece = get_double;
            break;
        case 8:
            if (strcmp(ctx->colnames->root->data, TIME_COLUMN)) {
                pcsv_errno = PCSV_ERR_DATA;
                return -1;
            }
            // no break
        default:
            ctx->status = STATUS_DATA;
            if (n_pieces != ctx->colnames->length) {
                pcsv_errno = PCSV_ERR_STRUCTURE;
                return -1;
            }
            break;
    }
    ctx->current_col = 0;
    return 0;
}

/*
 *
 *            Main rotines declared in pto_csv.h
 *
 */

struct pto_context *pcsv_new(void) {
    struct pto_context *ctx;
    static struct pto_context zero;
    if (!(ctx = malloc(sizeof *ctx))) return NULL;
    *ctx = zero;

    if (!(ctx->header_keys = list_new()))      goto CLEANUP;
    if (!(ctx->header_values = list_new()))    goto CLEANUP;
    if (!(ctx->colnames = list_new()))         goto CLEANUP;
    goto CONTINUE;
CLEANUP:
    pcsv_errno = PCSV_ERR_MALLOC;
    pcsv_free(ctx);
    return NULL;
CONTINUE:

    ctx->csv = csv_new(',', get_text, get_line, ctx);

    return ctx;
}

int pcsv_feed(struct pto_context *ctx, const char *buffer, int buflen) {
    if (ctx->status != STATUS_FEED) {
        ctx->status = STATUS_FEED;
    }
    return csv_feed(ctx->csv, buffer, buflen);
}

void pcsv_free(struct pto_context *ctx) {
    if (!ctx) return;
    list_free(ctx->header_keys);
    list_free(ctx->header_values);
    list_free(ctx->colnames);
    double_a_free(ctx->data);
    csv_free(ctx->csv);
    free(ctx->column_mask);
    free(ctx);
}

/*
void print_data(struct pto_context *ctx) {
    int c, i;
    printf("Header Keys:\n");
    ll_print(ctx->header_keys);
    printf("\nHeader Values:\n");
    ll_print(ctx->header_values);
    printf("\nColumns:\n");
    ll_print(ctx->colnames);
    for (c = 0; c < ctx->n_columns; ++c) {
        if (ctx->column_mask[c]) {
            printf("Column %d:\n", c);
            for (i = 0; i < ctx->data[c]->length; ++i) {
                printf(" %d: %f\n", i, ctx->data[c]->value[i]);
            }
        }
    }
}
*/
