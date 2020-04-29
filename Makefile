CFLAGS=-Wall -pedantic

.PHONY: all
all: test

test: test.o
	g++ -o $@ $^

%.o: %.cc gpu_gram_scan.h
	g++ $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f test *.o
