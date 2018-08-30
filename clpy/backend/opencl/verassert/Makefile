PROG=verassert

CXXFLAGS=-O2 -std=c++11 -Wall -Wextra -Wpedantic
LDFLAGS=-lOpenCL

all: $(PROG)

$(PROG): $(PROG).o
	$(CXX) -o $@ $< $(LDFLAGS)

run: $(PROG)
	./$(PROG) 1.2

clean:
	-rm $(PROG)
	-rm *.o
