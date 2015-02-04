CXX = g++
CXXFLAGS = -O3 -fopenmp -ffast-math -march=native
RM = rm
CCLIBDIRS = /home/jwillis/root/gcc/lib64

transpose:
	$(CXX) -o transpose.o -c $(CXXFLAGS) -fPIC Transpose.cc
	$(CXX) -o libtranspose.so -shared -fopenmp -lgomp -Wl,-rpath=$(CCLIBDIRS) transpose.o 

clean: 
	rm -f *.o *.so
