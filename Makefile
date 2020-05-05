CC=g++
CFLAGS=-std=c++17 -Wc++11-extensions -Wall -Wextra -pedantic

.PHONY: all
all: test

test: test.o
	$(CC) -o $@ $^
	find . -name '*.h' -o -name '*.cc' | xargs clang-format -i -style=Google

%.o: %.cc gpu_gram_scan.h
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f test *.o
