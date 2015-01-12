all:
	g++ -mavx2 -O3 -std=c++11 main.cpp
	g++ -mavx2 -O3 -std=c++11 main.cpp -S
