

def interpret_16b_as_twos_complement(int_value):
    return int_value - int((int_value << 1) & 65536)  # 65536 = 2**16


if __name__ == "__main__":
    sample = b'\x81\x11\xe6\xa4\xeb\xef\xa5\xdbr\x8e\x8a\xd9\xf8\xec5z\x94\xfd\x8a\xf8\xd5\x85\xc0'
    x = (sample[0] << 8) + sample[1]
    print(x, interpret_16b_as_twos_complement(x))
