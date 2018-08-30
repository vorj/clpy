PROG=verassert

CXXFLAGS=-O2 -std=c++11 -Wall -Wextra -Wpedantic
LDFLAGS=-lOpenCL

all: $(PROG)

$(PROG): $(PROG).o
	$(CXX) -o $@ $< $(LDFLAGS)

run: $(PROG)
	./$(PROG) ${CL_VERSION} ${DEVICE_INDEX}

clean:
	-rm $(PROG)
	-rm *.o
