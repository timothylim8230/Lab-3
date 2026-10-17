# FIT3143 Lab 3 - EV charging WSN simulation
#
#   make            -> wsn      (COMM_VERSION=2, non-blocking)
#   make all        -> wsn wsn_v1 wsn_v3
#   make debug      -> wsn_dbg  (-O0 -g -fsanitize=address,undefined)
#   make run        -> quick 3x3 smoke test
#   make clean

MPICC   ?= mpicc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -fopenmp
LDFLAGS ?= -fopenmp -lm
SRC      = src/main.c src/common.c src/node.c src/base.c
HDR      = src/common.h src/node.h src/base.h

.PHONY: default all debug run clean

default: wsn

all: wsn wsn_v1 wsn_v3

wsn: $(SRC) $(HDR)
	$(MPICC) $(CFLAGS) -DCOMM_VERSION=2 -o $@ $(SRC) $(LDFLAGS)

wsn_v1: $(SRC) $(HDR)
	$(MPICC) $(CFLAGS) -DCOMM_VERSION=1 -o $@ $(SRC) $(LDFLAGS)

wsn_v3: $(SRC) $(HDR)
	$(MPICC) $(CFLAGS) -DCOMM_VERSION=3 -o $@ $(SRC) $(LDFLAGS)

debug: $(SRC) $(HDR)
	$(MPICC) -std=c11 -O0 -g -Wall -Wextra -fopenmp -fsanitize=address,undefined -DCOMM_VERSION=2 -o wsn_dbg $(SRC) $(LDFLAGS)

# 3x3 grid, 6 ports, threshold 2, 2 s cycle, T = 10 s, run for 20 s, seed 1, high congestion
run: wsn
	mkdir -p logs
	mpirun -np 10 --oversubscribe ./wsn 3 3 6 2 2 10 20 1 0.8

clean:
	rm -f wsn wsn_v1 wsn_v3 wsn_dbg
