
CXX=mpicxx
FLAGS=-O2 -std=c++17 -g -fsanitize=address

LIB_DIR=-L/home/zezhao/local/gtest/lib -L/home/zezhao/local/gcc/lib64/
LIBS=-lgtest -lasan

SRC=test.cpp
OBJ=$(SRC:%.cpp=build/%.o)
DEP=$(SRC:%.cpp=build/%.d)
EXE=build/main

DIR=build/

all: ${EXE}

info:
	@echo "CXX=${CXX}"
	@echo "FLAGS=${FLAGS}"
	@echo "SRC=${SRC}"
	@echo "DEP=${DEP}"

-include  ${DEP}

${DIR}:
	mkdir -p ${DIR}

${EXE}: ${OBJ} | ${DIR}
	${CXX} ${FLAGS} ${OBJ} ${LIB_DIR} ${LIBS} -o $@

build/%.o: %.cpp | ${DIR}
	${CXX} ${FLAGS} -MMD -c $< -o $@

clean:
	-rm -rf build

test:
	mpirun -n 4 ${EXE}

.PHONY: clean info test
