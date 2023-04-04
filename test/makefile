#!/bin/bash

CXX=mpicxx
FLAGS=-O2 -std=c++17

LIBS=-lgtest

SRC=test.cpp


all: test

info:
	echo "CXX=${CXX}"
	echo "FLAGS=${FLAGS}"
	echo "SRC=${SRC}"

test: test.cpp
	${CXX} ${FLAGS} ${SRC} -o $@ 2> cerr.log

clean:
	rm test

.PHONY: clean info
