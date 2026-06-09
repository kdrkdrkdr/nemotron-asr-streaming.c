CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic -ffast-math
LDFLAGS ?= -lm

TARGET = nemotron_asr
SRCS = main.c nemotron_asr.c nemotron_asr_audio.c nemotron_asr_encoder.c nemotron_asr_decoder.c nemotron_asr_model.c nemotron_asr_kernels.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean debug

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c nemotron_asr.h
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS = -O0 -g -std=c11 -Wall -Wextra -pedantic -fsanitize=address
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
