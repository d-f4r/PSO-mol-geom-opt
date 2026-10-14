# ---[ makefile temp ]-----------------------------------------

# ===[ options = intel, amd, none ]============================
arch :=  intel
debug := true
# =============================================================

# ===[ file vars ]=============================================
TARGET := executablename
COMMON_DIR ?=  # dir με τα wrappers
CC := gcc
SRCS := main.c $(COMMON_DIR)/universal.c read_write.c pso_run.c lj.c
# =============================================================

ifeq ($(arch), intel)
	# μάλλον θα χρειαστεί να αλλαχθούν και τα δύο επόμενα ανάλογα με το directory των βιβλιοθηκών
	lib_dir := /opt/intel/oneapi/mkl/latest
	LDFLAGS := -L$(lib_dir)/lib/intel64 -Wl,-rpath,$(lib_dir)/lib/intel64 -lmkl_rt -lm -lpthread -ldl
	SRCS += $(COMMON_DIR)/mkl-math.c
else ifeq ($(arch), amd)
	lib_dir := /opt/AMD/aocl/aocl-linux-gcc-5.3.0/gcc/MT
	LDFLAGS := -L$(lib_dir)/lib -Wl,-rpath,$(lib_dir)/lib -laoclutils -lflame -lblis-mt -lm
	SRCS += $(COMMON_DIR)/aocl-math.c
else
	lib_dir :=
	LDFLAGS := -lm
endif

ifeq ($(debug), true)
	CFLAGS := -march=native -fopenmp -std=gnu11 -Wall -Werror -Wno-unused-function -O0 -g -fsanitize=address -MMD
	LDFLAGS += -fsanitize=address
else
	CFLAGS := -O3 -march=native -fopenmp -std=gnu11 -Wall -MMD
endif

INCLUDES := -I$(lib_dir)/include -I$(COMMON_DIR)

OBJS := $(SRCS:.c=.o)
DEPS := $(SRCS:.c=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(OBJS) $(TARGET) $(DEPS)

test:
	echo ${arch}
