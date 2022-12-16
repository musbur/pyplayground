#ifndef CSV_READ_H
#define CSV_READ_H

#include <stdlib.h>

#define BUFFER_INCREMENT 1000
#define EOL '\n'
#define EOL_SUPPRESS '\r'

typedef int (func_piece_t)(const char *piece, long pos, void *user_data);
typedef int (func_line_t)(int line_no, int n_pieces, void *user_data);

struct csv_context {
    char *buffer;
    char *p_end;
    int buflen;
    int bufpos;
    int line_no;
    long file_pos;
    int n_pieces;
    int sep;
    func_piece_t *func_piece;
    func_line_t *func_line;
    void *user_data;
} ;

struct csv_context *csv_new(int sep, func_piece_t func_piece,
                            func_line_t func_line,
                            void *user_data);

void csv_free(struct csv_context *ctx);

int csv_feed(struct csv_context *ctx, const char *bytes, int nbytes);

#endif
