

SRC=test.cpp
OBJ=$(SRC:%.cpp=build/%.o)
DEP=$(SRC:%.cpp=build/%.d)
EXE=build/main

DIR=build/

include config.make

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
