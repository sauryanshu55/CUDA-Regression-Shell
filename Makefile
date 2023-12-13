CC = nvcc
CFLAGS = 

TARGET = shell
SRCS = regression.cu

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
