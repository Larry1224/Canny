all:canny.exe canny-p.exe
CC=g++
CFLAGS=-Wall -std=c++17 -g -march=native -fopenmp -O -ltbb
# CFLAGS= -std=c++11 -march=native -fopenmp -w 
INCLUDE=/usr/include/opencv4/
LIBS= -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lstdc++

test.exe: test.cpp
	$(CC) $(CFLAGS) $< -o $@ -I $(INCLUDE) $(LIBS)

canny.exe: canny.cpp
	$(CC) $(CFLAGS) $< -o $@ -I $(INCLUDE) $(LIBS)
canny-p.exe: canny-p.cpp
	$(CC) $(CFLAGS) $< -o $@ -I $(INCLUDE) $(LIBS)
clean:
	rm *.exe 

test: test.exe
	./test.exe >> log

canny: canny.exe canny-p.exe
	./canny.exe >> log 
	# ./canny-p.exe >>log
	# OMP_NUM_THREADS=2 ./canny-p.exe >>log
	# OMP_NUM_THREADS=4 ./canny-p.exe >>log
	# OMP_NUM_THREADS=8 ./canny-p.exe >>log
	# OMP_NUM_THREADS=16 ./canny-p.exe >>log
	# OMP_NUM_THREADS=32 ./canny-p.exe >>log
