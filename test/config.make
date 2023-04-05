#Compilation
CXX=mpicxx
FLAGS=-O2 -std=c++17 -g -fsanitize=address

#Link
GTEST_INC=
GTEST_LIB=

ASAN_LIB=

LIB_DIR=-L${GTEST_LIB} -L${ASAN_LIB}
LIBS=-lgtest -lasan
