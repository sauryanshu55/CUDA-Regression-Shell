CC = gcc
CFLAGS = -Wall -Werror -g

TARGET = shell
SRCS = shell.c 

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
