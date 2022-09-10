import os
import tempfile
from typing import List

from range_coder import RangeDecoder as _RD
from range_coder import RangeEncoder as _RE


def normalize_cdf(cdf: List) -> List[int]:
    """
    Example: 
        >>> [7, 8, 9] -> [0, 7, 8, 9]
        >>> [0, 1, 1, 3] -> [0, 1, 2, 3, 4]
        >>> [4, 4, 4, 9] -> [0, 4, 5, 6, 9]
    """
    cdf_ = [0]
    for c in cdf:
        cdf_.append(max(int(c), cdf_[-1]+1))
    return cdf_

class RangeEncoder(object):
    def __init__(self) -> None:
        self.re = None

    def __enter__(self) -> 'RangeEncoder':
        self.tmpf = tempfile.NamedTemporaryFile()
        self.re = _RE(self.tmpf.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.tmpf.close()
        if self.re:
            self.re.close()

    def encode(self, symbol, qcdf, is_normalized: bool=False) -> None:
        if not(self.re):
            raise ValueError('Range Encoder is not initialized!')
        if not(is_normalized):
            qcdf = normalize_cdf(qcdf)
        self.re.encode(symbol, qcdf)

    def get_byte_string(self) -> bytes:
        if not(self.re):
            raise ValueError('Range Encoder is not initialized!')
        self.re.close()
        self.re = None
        self.tmpf.seek(0)
        return self.tmpf.read()


class RangeDecoder(object):
    def __init__(self, byte_string: bytes) -> None:
        self.rd = None
        self.string = byte_string

    def __enter__(self) -> 'RangeDecoder':
        self.tmpd = tempfile.TemporaryDirectory()
        bit_path = os.path.join(self.tmpd.name, 'bin.pth')
        with open(bit_path, 'wb') as f:
            f.write(self.string)
        self.rd = _RD(bit_path)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.tmpd.cleanup()
        if self.rd:
            self.rd.close()

    def decode(self, num_symbol: int, qcdf: List, is_normalized: bool=False) -> List[int]:
        if not(self.rd):
            raise ValueError('Range Decoder is not initialized!')
        if not(is_normalized):
            qcdf = normalize_cdf(qcdf)
        return self.rd.decode(num_symbol, qcdf)
