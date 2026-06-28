UNAME_S := $(shell uname -s)

# Windows (MSYS2/MinGW shell): clang targeting the MSVC runtime builds a native
# .exe. The Win32 platform paths replace pthread/mmap, so no -lm/-lpthread and
# no -pedantic/-flto (windows.h is noisy under -pedantic).
NEMO_WINDOWS :=
ifneq (,$(findstring MINGW,$(UNAME_S)))
NEMO_WINDOWS := 1
endif
ifneq (,$(findstring MSYS,$(UNAME_S)))
NEMO_WINDOWS := 1
endif
ifneq (,$(findstring CYGWIN,$(UNAME_S)))
NEMO_WINDOWS := 1
endif

ifeq ($(NEMO_WINDOWS),1)
# GNU make predefines CC=cc (origin "default"), so ?= would not switch it.
# Only force clang when the user has not chosen a compiler themselves.
ifeq ($(origin CC),default)
CC := clang
endif
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -march=native -ffast-math -D_CRT_SECURE_NO_WARNINGS
LDFLAGS ?=
EXE := .exe
else
CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic -march=native -ffast-math -flto
LDFLAGS ?= -lm -lpthread -flto
EXE :=
endif

SYNTAX_ARM_TARGET ?= arm64-apple-macos13
SYNTAX_X86_TARGET ?= x86_64-apple-macos13
SYNTAX_CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic -ffast-math -fsyntax-only

TARGET = nemotron_asr$(EXE)
MIC_TARGET = nemotron_asr_mic$(EXE)
KERNEL_CHECK_TARGET = nemotron_kernel_check$(EXE)
KERNEL_BENCH_TARGET = nemotron_kernel_bench$(EXE)
RUNTIME_SRCS = nemotron_asr.c nemotron_asr_audio.c nemotron_asr_encoder.c nemotron_asr_decoder.c nemotron_asr_model.c nemotron_asr_kernels.c nemotron_asr_kernels_generic.c nemotron_asr_kernels_neon.c nemotron_asr_kernels_avx.c
SRCS = main.c $(RUNTIME_SRCS)
OBJS = $(SRCS:.c=.o)
RUNTIME_OBJS = $(RUNTIME_SRCS:.c=.o)
MIC_OBJS = mic.o $(RUNTIME_OBJS)
KERNEL_CHECK_OBJS = kernel_check.o nemotron_asr_kernels_generic.o nemotron_asr_kernels_neon.o nemotron_asr_kernels_avx.o
KERNEL_BENCH_OBJS = kernel_bench.o nemotron_asr_kernels.o nemotron_asr_kernels_generic.o nemotron_asr_kernels_neon.o nemotron_asr_kernels_avx.o

.PHONY: all clean debug generic mic check-kernels bench-kernels check-arch-syntax

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(KERNEL_CHECK_TARGET): $(KERNEL_CHECK_OBJS)
	$(CC) $(CFLAGS) -o $@ $(KERNEL_CHECK_OBJS) $(LDFLAGS)

$(KERNEL_BENCH_TARGET): $(KERNEL_BENCH_OBJS)
	$(CC) $(CFLAGS) -o $@ $(KERNEL_BENCH_OBJS) $(LDFLAGS)

check-kernels: $(KERNEL_CHECK_TARGET)
	./$(KERNEL_CHECK_TARGET)

bench-kernels: $(KERNEL_BENCH_TARGET)
	./$(KERNEL_BENCH_TARGET)

check-arch-syntax:
	$(CC) -target $(SYNTAX_ARM_TARGET) $(SYNTAX_CFLAGS) nemotron_asr_kernels_neon.c
	$(CC) -target $(SYNTAX_X86_TARGET) $(SYNTAX_CFLAGS) -mavx2 -mfma nemotron_asr_kernels_avx.c

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

clean:
	rm -f $(OBJS) kernel_check.o kernel_bench.o mic.o $(KERNEL_BENCH_TARGET) $(KERNEL_CHECK_TARGET) $(MIC_TARGET) $(TARGET)
