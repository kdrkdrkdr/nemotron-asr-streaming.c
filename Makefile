UNAME_S := $(shell uname -s)

# Windows (MSYS2/MinGW shell): clang targeting the MSVC runtime builds a native
# .exe. The Win32 platform paths replace pthread/mmap, so no -lm/-lpthread and
# no -pedantic (windows.h is noisy under it); -flto stays POSIX-only here. These
# platform deltas live in PLAT_*/PEDANTIC so every build variant composes with
# the right base instead of hardcoding POSIX flags.
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
PLAT_DEFS   := -D_CRT_SECURE_NO_WARNINGS
PLAT_LDLIBS :=
PLAT_LTO    :=
PEDANTIC    :=
SANITIZE    :=
EXE         := .exe
else
CC ?= cc
PLAT_DEFS   :=
PLAT_LDLIBS := -lm -lpthread
PLAT_LTO    := -flto
PEDANTIC    := -pedantic
SANITIZE    := -fsanitize=address
EXE         :=
endif

CFLAGS  ?= -O3 -std=c11 -Wall -Wextra $(PEDANTIC) -march=native -ffast-math $(PLAT_LTO) $(PLAT_DEFS)
LDFLAGS ?= $(PLAT_LDLIBS) $(PLAT_LTO)

TARGET = nemotron_asr$(EXE)
SRCS = main.c nemotron_asr.c nemotron_asr_audio.c nemotron_asr_encoder.c nemotron_asr_decoder.c nemotron_asr_model.c nemotron_asr_kernels.c nemotron_asr_kernels_generic.c nemotron_asr_kernels_neon.c nemotron_asr_kernels_avx.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean debug generic

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c nemotron_asr.h nemotron_asr_kernels.h nemotron_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

# debug/generic reuse the platform deltas so they build on Windows too.
# SANITIZE is empty on Windows (clang's ASan needs its dynamic runtime DLL on
# PATH to run), so `make debug` is a plain -O0 -g build there and ASan-enabled
# on POSIX.
debug: CFLAGS  := -O0 -g -std=c11 -Wall -Wextra $(PEDANTIC) $(SANITIZE) $(PLAT_DEFS)
debug: LDFLAGS := $(PLAT_LDLIBS) $(SANITIZE)
debug: clean $(TARGET)

generic: CFLAGS  := -O3 -std=c11 -Wall -Wextra $(PEDANTIC) -ffast-math -DNEMO_FORCE_GENERIC $(PLAT_DEFS)
generic: LDFLAGS := $(PLAT_LDLIBS)
generic: clean $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
