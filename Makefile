CC ?= clang
CFLAGS ?= -O3 -Wall -Wextra
LIBS ?= -lm
TARGET = luminous
PREFIX ?= /usr/local

all: $(TARGET)

$(TARGET): src/lum_runtime.c
	$(CC) $(CFLAGS) src/lum_runtime.c $(LIBS) -o $(TARGET)

install: $(TARGET)
	mkdir -p $(DESTDIR)$(PREFIX)/bin
	cp -f $(TARGET) $(DESTDIR)$(PREFIX)/bin/
	chmod 755 $(DESTDIR)$(PREFIX)/bin/$(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all install clean
