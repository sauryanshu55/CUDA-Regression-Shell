CC = nvcc
CFLAGS = 

TARGET = shell
SRCS = shell.cu 

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
