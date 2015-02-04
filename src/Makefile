CC = gcc
RM = rm
CCLIBDIRS = /home/jwillis/root/gcc/lib64

libthresh: thresh.c
	$(CC) -o thresh.o -c -fPIC -fopenmp -march=native -O3 -funroll-loops thresh.c
	$(CC) -o libthresh.so -shared -fopenmp thresh.o -lgomp

libcorr: corr.c
	$(CC) -o corr.o -c -fPIC -fopenmp -march=native -O3 -Wall corr.c
	$(CC) -o libcorr.so -pthread -shared -fopenmp corr.o -lgomp

libncorr: ncorr.c
	$(CC) -o ncorr.o -c -fPIC -march=native -O3 -Wall ncorr.c
	$(CC) -o libncorr.so  -shared ncorr.o 

libscorr: simd_corr.c
	$(CC) -o simd_corr.o -c -fPIC -march=native -O3 -Wall simd_corr.c
	$(CC) -o libscorr.so -shared simd_corr.o

libacorr: acorr.c
	$(CC) -o acorr.o -c -fPIC -march=native -O2 -Wall acorr.c
	$(CC) -o libacorr.so -shared acorr.o

libtrans: trans.c
	$(CC) -o trans.o -c -fPIC -fopenmp -march=native -O3 -funroll-loops -ffast-math -Wall trans.c
	$(CC) -o libtrans.so -shared -fopenmp trans.o -lgomp -Wl,-rpath=$(CCLIBDIRS)

clean:
	$(RM) -f *.o *.so *~ *.i *.s