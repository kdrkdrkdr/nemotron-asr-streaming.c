CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic -march=native -ffast-math -flto
LDFLAGS ?= -lm -flto

TARGET = nemotron_asr
SRCS = main.c nemotron_asr.c nemotron_asr_audio.c nemotron_asr_encoder.c nemotron_asr_decoder.c nemotron_asr_model.c nemotron_asr_kernels.c nemotron_asr_kernels_generic.c nemotron_asr_kernels_neon.c nemotron_asr_kernels_avx.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean debug generic

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c nemotron_asr.h nemotron_asr_kernels.h nemotron_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS = -O0 -g -std=c11 -Wall -Wextra -pedantic -fsanitize=address
debug: LDFLAGS = -lm -fsanitize=address
debug: clean $(TARGET)

generic: CFLAGS = -O3 -std=c11 -Wall -Wextra -pedantic -ffast-math -DNEMO_FORCE_GENERIC
generic: LDFLAGS = -lm
generic: clean $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
