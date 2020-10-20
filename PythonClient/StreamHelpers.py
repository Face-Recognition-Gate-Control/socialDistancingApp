
import struct
from typing import List, Type, Callable, NewType

# The order of the bytes from the stream little or big
BYTE_ORDER = "big"
# Encoding for strings
ENCODING = "UTF-8"


# Reads a single byte from a stream, and return the value of the byte
def read_byte(reader: Callable, signed: bool = False):
    return int.from_bytes(
        reader(1), byteorder=BYTE_ORDER, signed=signed)

# Reads a single short from a stream, and return the value of the short


def read_short(reader: Callable, signed: bool = False):
    return int.from_bytes(
        reader(2), byteorder=BYTE_ORDER, signed=signed)

# Reads a single int from a stream, and return the value of the int


def read_int(reader: Callable, signed: bool = False):
    return int.from_bytes(
        reader(4), byteorder=BYTE_ORDER, signed=signed)

# Reads a single long from a stream, and return the value of the long


def read_long(reader: Callable, signed: bool = False):
    return int.from_bytes(
        reader(8), byteorder=BYTE_ORDER, signed=signed)

# Reads a single float from a stream, and return the value of the float


def read_float(reader: Callable):
    struct.unpack('f', reader(4))

# Reads a single byte from a stream, and return the value of the byte


def read_double(reader: Callable):
    struct.unpack('f', reader(8))

# Reads n bytes from stream and convert it to a string.


def read_bytes_to_string(reader: Callable, nBytes):
    return reader(nBytes).decode(ENCODING)
