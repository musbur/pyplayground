
test_pto_csv: test_pto_csv.c csv_read.o pto_csv.o
	gcc -O0 -W -Wall -g -fmax-errors=1 -o $@ $^

%.o: %.c %.h
	gcc -W -Wall -g -c -o $@ $<

test: test_pto_csv
	./test_pto_csv

clean:
	rm -f *.o test_pto_csv
