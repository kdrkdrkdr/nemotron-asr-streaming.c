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

TARGET = nemotron_asr$(EXE)
SRCS = main.c nemotron_asr.c nemotron_asr_audio.c nemotron_asr_encoder.c nemotron_asr_decoder.c nemotron_asr_model.c nemotron_asr_kernels.c nemotron_asr_kernels_generic.c nemotron_asr_kernels_neon.c nemotron_asr_kernels_avx.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean debug generic

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c nemotron_asr.h nemotron_asr_kernels.h nemotron_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS = -O0 -g -std=c11 -Wall -Wextra -pedantic -fsanitize=address
debug: LDFLAGS = -lm -lpthread -fsanitize=address
debug: clean $(TARGET)

generic: CFLAGS = -O3 -std=c11 -Wall -Wextra -pedantic -ffast-math -DNEMO_FORCE_GENERIC
generic: LDFLAGS = -lm -lpthread
generic: clean $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
