CC = gcc
CFLAGS = -g 
LDFLAGS = -lOpenCL -lm -I/usr/local/cuda/include
LDTEST = -lfftw3 

REQ = src/util.c src/checkers.c test/float1D.c 
OBJ = obj/util.o obj/checkers.o obj/float1D.o

topeFFT: $(REQ)
	@$(CC) $(CFLAGS) -c -o obj/util.o src/util.c $(LDFLAGS) $(LDTEST)
	@$(CC) $(CFLAGS) -c -o obj/float1D.o test/float1D.c $(LDFLAGS) $(LDTEST)
	@$(CC) $(CFLAGS) -c -o obj/checkers.o src/checkers.c $(LDFLAGS)
	@$(CC) $(OBJ) $(LDFLAGS) $(LDTEST)

clean:
	@rm -f obj/*.o
	@rm -f a.out
	@rm -f .sw*
	@rm -f .*sw*
	@rm -f src/.sw*
