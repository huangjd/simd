CC := g++
CFLAGS := -mavx2 -std=c++11

ifeq ($(DEBUG), 1)
	CFLAGS += -O0 -g
else
	CFLAGS += -O3 -DNDEBUG
endif

all:
	rm a.out
	rm *.s
	$(CC) $(CFLAGS) main.cpp
	$(CC) $(CFLAGS) -S main.cpp
