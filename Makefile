CFLAGS=-Wall -Wextra -pedantic

.PHONY: all
all: test

test: test.o
	g++ -o $@ $^
	find . -name '*.h' -o -name '*.cc' | xargs clang-format -i -style=Google

%.o: %.cc gpu_gram_scan.h
	g++ $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f test *.o
