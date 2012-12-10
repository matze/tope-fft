CC = gcc
CFLAGS = -Wall -fPIC -O2 
LDFLAGS = -lOpenCL -lm -I/usr/local/cuda/include 

REQ = src/util.c src/fft1d.c src/fft3d.c src/fft2d.c 
OBJ = obj/util.o obj/fft1d.o obj/fft3d.o obj/fft2d.o 

topeFFT: $(REQ)
	mkdir -p obj
	mkdir -p lib
	$(CC) $(CFLAGS) -c -o obj/util.o src/util.c $(LDFLAGS) 
	$(CC) $(CFLAGS) -c -o obj/fft1d.o src/fft1d.c $(LDFLAGS)
	$(CC) $(CFLAGS) -c -o obj/fft2d.o src/fft2d.c $(LDFLAGS)
	$(CC) $(CFLAGS) -c -o obj/fft3d.o src/fft3d.c $(LDFLAGS)
	$(CC) -shared -Wl,-soname,libTopeFFT.so.1 -o lib/libTopeFFT.so.1.0 $(OBJ)

install:
	mkdir -p /opt/topefft
	cp src/kernels1D.cl /opt/topefft
	cp src/kernels2D.cl /opt/topefft
	cp src/kernels3D.cl /opt/topefft
	cp lib/libTopeFFT.so.1.0 /opt/topefft/libTopeFFT.so.1.0
	ln -sf /opt/topefft/libTopeFFT.so.1.0 /opt/topefft/libTopeFFT.so.1
	ln -sf /opt/topefft/libTopeFFT.so.1.0 /opt/topefft/libTopeFFT.so

clean:
	@rm -f obj/*.o
	@rm -f bin/*
	@rm -f a.out
	@rm -f .sw*
	@rm -f .*sw*
	@rm -f src/.sw*

