CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic -march=native -ffast-math -flto
LDFLAGS ?= -lm -lpthread -flto
SYNTAX_ARM_TARGET ?= arm64-apple-macos13
SYNTAX_X86_TARGET ?= x86_64-apple-macos13
SYNTAX_CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic -ffast-math -fsyntax-only

TARGET = nemotron_asr
MIC_TARGET = nemotron_asr_mic
KERNEL_CHECK_TARGET = nemotron_kernel_check
RUNTIME_SRCS = nemotron_asr.c nemotron_asr_audio.c nemotron_asr_encoder.c nemotron_asr_decoder.c nemotron_asr_model.c nemotron_asr_kernels.c nemotron_asr_kernels_generic.c nemotron_asr_kernels_neon.c nemotron_asr_kernels_avx.c
SRCS = main.c $(RUNTIME_SRCS)
OBJS = $(SRCS:.c=.o)
RUNTIME_OBJS = $(RUNTIME_SRCS:.c=.o)
MIC_OBJS = mic.o $(RUNTIME_OBJS)
KERNEL_CHECK_OBJS = kernel_check.o nemotron_asr_kernels_generic.o nemotron_asr_kernels_neon.o nemotron_asr_kernels_avx.o
UNAME_S := $(shell uname -s)

.PHONY: all clean debug generic blas mic check-kernels check-arch-syntax

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(KERNEL_CHECK_TARGET): $(KERNEL_CHECK_OBJS)
	$(CC) $(CFLAGS) -o $@ $(KERNEL_CHECK_OBJS) $(LDFLAGS)

check-kernels: $(KERNEL_CHECK_TARGET)
	./$(KERNEL_CHECK_TARGET)

check-arch-syntax:
	$(CC) -target $(SYNTAX_ARM_TARGET) $(SYNTAX_CFLAGS) nemotron_asr_kernels_neon.c
	$(CC) -target $(SYNTAX_X86_TARGET) $(SYNTAX_CFLAGS) -mavx2 -mfma nemotron_asr_kernels_avx.c
	$(CC) -target $(SYNTAX_X86_TARGET) $(SYNTAX_CFLAGS) -mavx2 -mfma -mavxvnni nemotron_asr_kernels_avx.c
	$(CC) -target $(SYNTAX_X86_TARGET) $(SYNTAX_CFLAGS) -mavx2 -mfma -mavxvnniint8 nemotron_asr_kernels_avx.c
	$(CC) -target $(SYNTAX_X86_TARGET) $(SYNTAX_CFLAGS) -mavx2 -mfma -mavx512f -mavx512bw -mavx512vnni nemotron_asr_kernels_avx.c

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
	rm -f $(OBJS) kernel_check.o mic.o $(KERNEL_CHECK_TARGET) $(MIC_TARGET) $(TARGET)
