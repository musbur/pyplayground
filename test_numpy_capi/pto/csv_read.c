#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "csv_read.h"

struct csv_context *csv_new(int sep, func_piece_t func_piece,
                            func_line_t func_line,
                            void *user_data) {
    struct csv_context *ctx;
    static struct csv_context zero;
    if (!(ctx = malloc(sizeof *ctx))) return NULL;
    *ctx = zero;
    ctx->line_no = 1;
    ctx->sep = sep;
    ctx->func_piece = func_piece;
    ctx->func_line = func_line;
    ctx->user_data = user_data;
    return ctx;
}

void csv_free(struct csv_context *ctx) {
    free(ctx->buffer);
    free(ctx);
}

int csv_feed(struct csv_context *ctx, const char *bytes, int nbytes) {
    int r = 0;
    const char *p;
    const char *ep = bytes + nbytes;
    for (p = bytes; p < ep; ++p) {
        if (ctx->bufpos == ctx->buflen) {
            char *t;
            ctx->buflen += BUFFER_INCREMENT;
            t = realloc(ctx->buffer, ctx->buflen + 1);
            if (t) ctx->buffer = t;
            else return -1;
            ctx->p_end = ctx->buffer + ctx->bufpos;
        }
        if (*p == ctx->sep || *p == EOL) {
            *ctx->p_end = 0;
            r = ctx->func_piece(ctx->buffer, ctx->file_pos, ctx->user_data);
            ++ctx->n_pieces;
            if (r != 0) return r;
            if (*p == EOL) {
                r = ctx->func_line(ctx->line_no, ctx->n_pieces, ctx->user_data);
                if (r != 0) return r;
                ++ctx->line_no;
                ctx->n_pieces= 0;
            }
            ctx->bufpos = 0;
            ctx->p_end = ctx->buffer;
        } else if (*p != EOL_SUPPRESS) {
            *(ctx->p_end++) = *p;
        }
        ++ctx->file_pos;
        ++ctx->bufpos;
    }
    return 0;
}

