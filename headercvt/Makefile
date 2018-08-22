build: headercvt

headercvt: headercvt.cpp
	clang++ -std=c++11 -Wall -Wextra -O3 -pedantic-errors $< -lclangTooling -lclangFrontendTool -lclangFrontend -lclangDriver -lclangSerialization -lclangCodeGen -lclangParse -lclangSema -lclangStaticAnalyzerFrontend -lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore -lclangAnalysis -lclangEdit -lclangAST -lclangLex -lclangBasic -lclang `llvm-config --libs --system-libs` -fno-rtti -o $@



%.pxi: headercvt
	./headercvt stub.c -- $(CLPY_HEADERCVT_INCLUDE_DIRS)



DEPLOY_PATH=../clpy/backend/opencl

$(DEPLOY_PATH)/%.pxi: %.pxi
	cp -f $< $@

deploy: \
	$(DEPLOY_PATH)/func_decl.pxi \
	$(DEPLOY_PATH)/preprocessor_defines.pxi \
	$(DEPLOY_PATH)/types.pxi


clean:
	rm headercvt 2>/dev/null || true
	rm *.pxi  2>/dev/null || true
