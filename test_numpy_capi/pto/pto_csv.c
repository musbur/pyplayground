#define _XOPEN_SOURCE

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "pto_csv.h"

#define STATUS_START 0
#define STATUS_FEED 2
#define STATUS_DATA 3

/*
 *
 *            Some helper data structures
 *
 */

struct ll {
    struct ll *next;
    char *s;
} ;

void ll_print(struct ll *ll) {
    int i = 1;
    while (ll && ll->next) {
        printf("  %d: <%s>\n", i++, ll->s);
        ll = ll->next;
    }
}

struct ll *ll_new(void) {
    struct ll *e;
    if (!(e = malloc(sizeof *e))) return NULL;
    e->s = NULL;
    e->next = NULL;
    return e;
}

struct ll *ll_append(struct ll *e, const char *s) {
    if (!(e->s = malloc(strlen(s) + 1))) return NULL;
    strcpy(e->s, s);
    e->next = ll_new();
    return e->next;
}

void ll_free(struct ll *ll) {
    struct ll *t;
    while (ll) {
        free(ll->s);
        t = ll->next;
        free(ll);
        ll = t;
    }
}

struct double_array {
    double *value;
    double *pos;
    int length;
    int alloc_increment;
    int allocated;
} ;

struct double_array *double_a_new(int alloc_increment) {
    struct double_array *arr;
    static struct double_array zero;
    if (!(arr = malloc(sizeof *arr))) return NULL;
    *arr = zero;
    arr->alloc_increment = alloc_increment;
    return arr;
}

int double_a_append(struct double_array *arr, double v) {
    if (arr->length == arr->allocated) {
        double *t;
        arr->allocated += arr->alloc_increment;
        t = realloc(arr->value, arr->allocated * sizeof *t);
        if (!t) return -1;
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
    pto_context *ctx = user_data;
    (void) pos;
    if (ctx->ll_p) ctx->ll_p = ll_append(ctx->ll_p, piece);
    return 0;
}

static int get_double(const char *piece, long pos, void *user_data) {
    pto_context *ctx = user_data;
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

static int endswith(const char *str, const char *pat) {
    // Returns nonzero if pat matches the end of str
    int diff = strlen(str) - strlen(pat);
    if (diff < 0) return 0;
    return (strcmp(pat, str + diff) == 0);
}

static char *make_colmask(struct ll *all, int n, char **wanted) {
    char *mask;
    char **wp;
    int i;
    struct ll *p = all;

    if (!(mask = malloc(n))) return NULL;
    for (i = 0; i < n; ++i) {
        mask[i] = 0;
        if (all) {
            for (wp = wanted; *wp; ++wp) {
                if (endswith(p->s, *wp)) {
                    mask[i] = 1;
                    break;
                }
            }
        } else {
            mask[i] = 1;
        }
        p = p->next;
        if (!p) break;
    }
    return mask;
}

static int get_line(int line_no, int n_pieces, void *user_data) {
    pto_context *ctx = user_data;
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
            ctx->column_mask = make_colmask(ctx->colnames, ctx->n_columns,
                                            ctx->colname_patterns);
            ctx->data = double_a_new(300);
            ctx->csv->func_piece = get_double;
            break;
        default:
            ctx->status = STATUS_DATA;
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

/*
 *
 *            Main rotines declared in pto_csv.h
 *
 */

pto_context *pcsv_new(void) {
    pto_context *ctx;
    static pto_context zero;
    if (!(ctx = malloc(sizeof *ctx))) return NULL;
    *ctx = zero;

    if (!(ctx->header_keys = ll_new()))      goto CLEANUP;
    if (!(ctx->header_values = ll_new()))    goto CLEANUP;
    if (!(ctx->colnames = ll_new()))         goto CLEANUP;
    goto CONTINUE;
CLEANUP:
    pcsv_free(ctx);
    return NULL;
CONTINUE:

    ctx->csv = csv_new(',', get_text, get_line, ctx);

    return ctx;
}

int pcsv_feed(pto_context *ctx, const char *buffer, int buflen) {
    if (ctx->status != STATUS_FEED) {
        ctx->ll_p = NULL;
        ctx->status = STATUS_FEED;
    }
    return csv_feed(ctx->csv, buffer, buflen);
}

void pcsv_free(pto_context *ctx) {
    ll_free(ctx->header_keys);
    ll_free(ctx->header_values);
    ll_free(ctx->colnames);
    double_a_free(ctx->data);
    csv_free(ctx->csv);
    free(ctx->column_mask);
    free(ctx);
}

/*
void print_data(pto_context *ctx) {
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
