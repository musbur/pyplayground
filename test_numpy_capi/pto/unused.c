
typedef struct {
    char **member;
    int pos;
    int size;
} csv_stringarray ;

typedef struct {
    struct _ll_member *root;
    struct _ll_member *pos;
    int size;
} csv_stringlist;

csv_stringlist *csv_sl_new(void);

csv_stringarray *csv_sa_new(int init_size);










csv_stringarray *csv_sa_new(int size) {
    csv_stringarray *sa;
    if (!(sa = malloc(sizeof *sa))) return NULL;
    sa->size = size;
    sa->member = malloc((size + 1) * sizeof *sa->member);
    sa->pos = 0;
    *sa->member = NULL;
    return sa;
}

void csv_sa_free(csv_stringarray *sa, int free_members) {
    if (free_members) {
        int i;
        for (i = 0; i < sa->size; ++i) free(sa->member[i]);
    }
    free(sa);
}

int csv_sa_append(csv_stringarray *sa, char *new) {
    if (sa->pos == sa->size) return -1;
    sa->member[sa->pos++] = new;
    sa->member[sa->pos] = NULL;
    return sa->pos;
}

int csv_sa_append_alloc(csv_stringarray *sa, char *new) {
    char *cp;
    if (sa->pos >= sa->size) return -1;
    if (!(cp = malloc(strlen(new)))) return -2;
    return csv_sa_append(sa, cp);
}

int csv_sa_truncate(csv_stringarray *sa, int index) {
    if (index > sa->pos) return -2;
    sa->member[index] = NULL;
    sa->pos = index;
    return sa->pos;
}

int csv_sa_truncate_free(csv_stringarray *sa, int index) {
    int i;
    if (index > sa->pos) return -2;
    for (i = index; i < sa->size; ++i) free(sa->member[i]);
    return csv_sa_truncate(sa, index);
}

struct _ll_member {
    char *s;
    struct _ll_member *next;
} ;

csv_stringlist *csv_sl_new(void) {
    csv_stringlist *sl;
    if (!(sl = malloc(sizeof *sl))) return NULL;
}

int csv_stringlist_append(csv_stringlist *sl, char *new) {

}
