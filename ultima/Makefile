ultima: ultima.cpp
	clang++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=$(use_cxx11_abi) -Wall -Wextra -O3 -pedantic-errors $< -lclangTooling -lclangFrontendTool -lclangFrontend -lclangDriver -lclangSerialization -lclangCodeGen -lclangParse -lclangSema -lclangStaticAnalyzerFrontend -lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore -lclangAnalysis -lclangARCMigrate -lclangRewrite -lclangEdit -lclangAST -lclangLex -lclangBasic -lclang `llvm-config --libs --system-libs` -fno-rtti -o $@

clean:
	rm ultima 2>/dev/null || true
