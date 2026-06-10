CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic -march=native -ffast-math -flto
LDFLAGS ?= -lm -lpthread -flto

TARGET = nemotron_asr
MIC_TARGET = nemotron_asr_mic
RUNTIME_SRCS = nemotron_asr.c nemotron_asr_audio.c nemotron_asr_encoder.c nemotron_asr_decoder.c nemotron_asr_model.c nemotron_asr_kernels.c nemotron_asr_kernels_generic.c nemotron_asr_kernels_neon.c nemotron_asr_kernels_avx.c
SRCS = main.c $(RUNTIME_SRCS)
OBJS = $(SRCS:.c=.o)
RUNTIME_OBJS = $(RUNTIME_SRCS:.c=.o)
MIC_OBJS = mic.o $(RUNTIME_OBJS)
UNAME_S := $(shell uname -s)

.PHONY: all clean debug generic blas mic

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

ifeq ($(UNAME_S),Darwin)
mic: $(MIC_TARGET)

$(MIC_TARGET): $(MIC_OBJS)
	$(CC) $(CFLAGS) -o $@ $(MIC_OBJS) $(LDFLAGS) -framework AudioToolbox -framework CoreAudio -framework CoreFoundation
else
mic:
	@echo "nemotron_asr_mic is currently macOS-only."
	@false
endif

%.o: %.c nemotron_asr.h nemotron_asr_kernels.h nemotron_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS = -O0 -g -std=c11 -Wall -Wextra -pedantic -fsanitize=address
debug: LDFLAGS = -lm -lpthread -fsanitize=address
debug: clean $(TARGET)

generic: CFLAGS = -O3 -std=c11 -Wall -Wextra -pedantic -ffast-math -DNEMO_FORCE_GENERIC
generic: LDFLAGS = -lm -lpthread
generic: clean $(TARGET)

ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = -O3 -std=c11 -Wall -Wextra -pedantic -march=native -ffast-math -flto -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS = -lm -lpthread -flto -framework Accelerate
else
blas: CFLAGS = -O3 -std=c11 -Wall -Wextra -pedantic -march=native -ffast-math -flto -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS = -lm -lpthread -flto -lopenblas
endif
blas: clean $(TARGET)

clean:
	rm -f $(OBJS) mic.o $(MIC_TARGET) $(TARGET)
