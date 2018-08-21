all: headercvt

headercvt: headercvt.cpp
	clang++ -std=c++11 -Wall -Wextra -O3 -pedantic-errors $< -lclangTooling -lclangFrontendTool -lclangFrontend -lclangDriver -lclangSerialization -lclangCodeGen -lclangParse -lclangSema -lclangStaticAnalyzerFrontend -lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore -lclangAnalysis -lclangEdit -lclangAST -lclangLex -lclangBasic -lclang `llvm-config --libs --system-libs` -fno-rtti -o $@


run:
	./headercvt stub.c -- $(CLPY_HEADERCVT_INCLUDE_DIRS)
	cp -f *.pxi ../clpy/backend/opencl/

clean:
	rm headercvt 2>/dev/null || true
	rm *.pxi  2>/dev/null || true
