CFLAGS += -Wall -fPIC -O2

LIB_SRC = src/util.c src/fft1d.c src/fft3d.c src/fft2d.c
LIB_OBJ = $(patsubst %.c,%.o,$(LIB_SRC))
LIB_BASE = topefft
LIB_NAME = lib$(LIB_BASE).so
LIB_SONAME = $(LIB_NAME).1
LIB = $(LIB_SONAME).0
LIB_LDFLAGS = $(LDFLAGS) -lOpenCL -lm

BIN_SRC = src/topeFFT_cc.c
BIN_OBJ = $(patsubst %.c,%.o,$(BIN_SRC))
BIN = topeFFT_cc
BIN_LDFLAGS = $(LIB_LDFLAGS) -L. -l$(LIB_BASE)

all: $(BIN)

%.o: %.c
	@echo " CC $@"
	@$(CC) -c $(CFLAGS) -o $@ $<

$(LIB): $(LIB_OBJ)
	@echo " LD $@"
	@$(CC) -shared -Wl,-soname,$(SONAME) -o $@ $(OBJ) -o $@ $(LIB_LDFLAGS)
	ln -s $(LIB) $(LIB_SONAME)
	ln -s $(LIB_SONAME) $(LIB_NAME)

$(BIN): $(LIB) $(BIN_OBJ)
	@echo " LD $@"
	@$(CC) $(BIN_OBJ) -o $@ $(BIN_LDFLAGS)

clean:
	@rm -f $(BIN_OBJ) $(LIB_OBJ) $(BIN) $(LIB_NAME) $(LIB_SONAME) $(LIB)
