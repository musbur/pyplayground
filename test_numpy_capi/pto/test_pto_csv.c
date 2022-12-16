
#define _XOPEN_SOURCE

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "pto_csv.h"

void ll_print(struct ll *ll);

char *wanted_columns[] = {"TIME", "ActualPower", "ActualVoltage",
                          "ESCControl SV_actualCapacity", NULL};

int main(void) {
    FILE *csv;
    int bytes;
    char buffer[100];
    pto_context *context;
    int r;

    context = pcsv_new();
    context->colname_patterns = wanted_columns;
    csv = fopen("Process Details - WLN8R0  b.11 - SPM6 - 16-22-41.csv", "rt");
    while ((bytes = fread(buffer, 1, sizeof buffer, csv)) > 0) {
        r = pcsv_feed(context, buffer, bytes);
        if (r) break;
    }
    fclose(csv);

    pcsv_free(context);
    return 0;
}

