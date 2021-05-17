#
# Very simple makefile for coursework 2. You are free to update this if you
# wish to add files etc., but ensure (a) the executable name is unchanged,
# (b) it still works on a school machine, and (c) the file cwk2_extra.h is
# unaltered.
#
EXE = cwk2
CC = mpicc
CCFLAGS = -Wall -lm -std=c99

all:
	$(CC) $(CCFLAGS) -o $(EXE) cwk2.c

