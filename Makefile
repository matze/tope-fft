CC = gcc
CFLAGS = -Wall -fPIC -O2 -g
LDFLAGS = -lOpenCL -lm -I/usr/local/cuda/include
LDTEST = -lfftw3 -lOpenCL -lTopeFFT -L/opt/topefft

REQ = src/util.c src/checkers.c 
OBJ = obj/util.o obj/checkers.o

topeFFT: $(REQ)
	$(CC) $(CFLAGS) -c -o obj/util.o src/util.c $(LDFLAGS) 
	$(CC) $(CFLAGS) -c -o obj/checkers.o src/checkers.c $(LDFLAGS)
	$(CC) -shared -Wl,-soname,libTopeFFT.so.1 -o lib/libTopeFFT.so.1.0 $(OBJ)

tests:
	$(CC) test/1d.c -o bin/1d $(LDTEST)

install:
	mkdir -p /opt/topefft
	cp src/kernels.cl /opt/topefft
	mv lib/libTopeFFT.so.1.0 /opt/topefft/libTopeFFT.so.1.0
	ln -sf /opt/topefft/libTopeFFT.so.1.0 /opt/topefft/libTopeFFT.so.1
	ln -sf /opt/topefft/libTopeFFT.so.1.0 /opt/topefft/libTopeFFT.so

clean:
	@rm -f obj/*.o
	@rm -f bin/*
	@rm -f a.out
	@rm -f .sw*
	@rm -f .*sw*
	@rm -f src/.sw*

