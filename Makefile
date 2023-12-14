CC = nvcc
CFLAGS = 

TARGET = shell
SRCS = regression.cu

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) `gsl-config --cflags` -o $@ $^ `gsl-config --libs`

clean:
	rm -f $(TARGET)

