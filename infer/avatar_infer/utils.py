from typing import List


def tts_codes_to_token(codes: List[int]):
    bit_width = 12
    combined = 0
    for x in codes:
        combined = (combined << bit_width) | x
    return combined


def token_to_tts_codes(token: int, code_size: int):
    bit_width = 12
    mask = (1 << bit_width) - 1  
    vals = []
    for _ in range(code_size):
        x = token & mask
        token >>= bit_width
        vals.append(x)
    vals.reverse()
    return vals
