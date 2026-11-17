"""numpy-only horizontal-shift proxy (phase correlation on the central half of rows at 256x256)."""
import numpy as np
from PIL import Image
_win = None
def gray256(img):
    return np.asarray(img.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float64)
def pc_shift(a, b):
    """return (dx, dy): content in b is displaced by (+dx right, +dy down) relative to a."""
    global _win
    A = a[64:192]; B = b[64:192]
    if _win is None or _win.shape != A.shape:
        _win = np.outer(np.hanning(A.shape[0]), np.hanning(A.shape[1]))
    A = (A - A.mean()) * _win; B = (B - B.mean()) * _win
    FA, FB = np.fft.fft2(A), np.fft.fft2(B)
    R = FB * np.conj(FA); R /= np.abs(R) + 1e-9
    r = np.real(np.fft.ifft2(R))
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    H, W = r.shape
    def sub(c, i, n, axis):
        if axis == 1: m, z, p = r[iy, (i-1) % n], r[iy, i], r[iy, (i+1) % n]
        else: m, z, p = r[(i-1) % n, ix], r[i, ix], r[(i+1) % n, ix]
        den = m - 2*z + p
        off = 0.5 * (m - p) / den if abs(den) > 1e-12 else 0.0
        v = i + off
        return v - n if v > n / 2 else v
    return sub(None, ix, W, 1), sub(None, iy, H, 0)
