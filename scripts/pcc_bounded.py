"""Phase cross-correlation with a bounded search window.

The stock skimage phase_cross_correlation takes the global maximum of the
correlation surface.  On weak, hazy data that surface has several comparable
peaks and the global max can jump between them from one frame to the next --
which shows up as an instantaneous "teleport" in the trajectory that no real
sample can perform.

Bounding the search to a physically plausible neighbourhood of a predicted
shift removes that failure mode without changing anything else about the
method.
"""
import numpy as np


def _integer_peak(ref, mov, max_shift, predicted, xp):
    """Integer-pixel shift from a phase-correlation surface, search bounded."""
    F1 = xp.fft.fftn(ref)
    F2 = xp.fft.fftn(mov)
    R = F1 * xp.conj(F2)
    R /= xp.maximum(xp.abs(R), 1e-12)          # phase normalisation
    cc = xp.real(xp.fft.ifftn(R))
    cc = xp.fft.fftshift(cc)                   # zero shift now at the centre

    shape = tuple(ref.shape)
    nd = ref.ndim
    mid = np.array([s // 2 for s in shape])

    if max_shift is not None:
        pred = np.zeros(nd) if predicted is None else np.asarray(predicted, float)
        neg_inf = xp.asarray(-np.inf, dtype=cc.dtype)
        for ax in range(nd):
            n = shape[ax]
            coord = xp.arange(n) - int(mid[ax])          # candidate shift
            ok = (coord >= pred[ax] - max_shift[ax]) & \
                 (coord <= pred[ax] + max_shift[ax])
            sh = [1] * nd
            sh[ax] = n
            cc = xp.where(ok.reshape(sh), cc, neg_inf)

    peak = np.array(np.unravel_index(int(xp.argmax(cc)), shape), dtype=float)
    return peak - mid


def pcc_bounded(ref, mov, max_shift=None, predicted=None, xp=None,
                upsample=10):
    """Shift to apply to `mov` (via ndimage.shift) to align it with `ref`.

    Two steps: a search-bounded integer peak, then skimage's upsampled-DFT
    refinement on the integer-corrected image.  The bound removes the
    peak-hopping failure; the refinement supplies the sub-pixel accuracy.

    Parameters
    ----------
    ref, mov : ndarray, same shape, any dimensionality
    max_shift : sequence or None
        Per-axis search radius in pixels.  None = unbounded (stock behaviour).
    predicted : sequence or None
        Centre of the search window, in pixels.  None = zero.
    """
    from skimage.registration import phase_cross_correlation

    if xp is None:
        xp = np
    ref = xp.asarray(ref, dtype=xp.float32)
    mov = xp.asarray(mov, dtype=xp.float32)

    coarse = _integer_peak(ref, mov, max_shift, predicted, xp)

    if upsample and upsample > 1:
        # np.roll is exact -- no interpolation error enters the residual
        rolled = xp.roll(mov, tuple(int(v) for v in coarse),
                         axis=tuple(range(ref.ndim)))
        r_h = ref if xp is np else xp.asnumpy(ref)
        m_h = rolled if xp is np else xp.asnumpy(rolled)
        # NOTE normalization=None here on purpose.  Full phase whitening makes
        # the correlation surface a near-delta, and the upsampled DFT then has
        # nothing sub-pixel left to find -- it returns exactly 0.  Plain
        # cross-correlation keeps the peak shape that carries the sub-pixel
        # information.  Phase whitening is still what located the integer peak,
        # where its robustness is what matters.
        out = phase_cross_correlation(r_h, m_h, upsample_factor=upsample,
                                      normalization=None)
        resid = np.asarray(out[0] if isinstance(out, tuple) else out, float)
        # the residual must be sub-pixel; if it is not, the bounded peak
        # already had it and we keep the integer answer
        if np.abs(resid).max() <= 1.5:
            return coarse + resid
    return coarse


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from scipy import ndimage as ndi

    rng = np.random.default_rng(0)
    base = ndi.gaussian_filter(rng.random((20, 128, 128)).astype(np.float32),
                               (1.0, 4.0, 4.0))

    print("sign / accuracy check -- integer shifts (np.roll, exact)")
    ok = True
    for true in [(0, 0, 0), (0, 7, -11), (2, -13, 5), (-1, 21, 17)]:
        # displace base by -true; recovering `true` puts it back
        moved = np.roll(base, tuple(-t for t in true), axis=(0, 1, 2))
        got = pcc_bounded(base, moved)
        err = np.abs(np.array(got) - np.array(true))
        good = err.max() < 0.15
        ok &= good
        print(f"  true={true}  recovered={np.round(got, 3)}  "
              f"{'OK' if good else 'MISMATCH'}")

    print("\nsub-pixel accuracy (fractional shifts, cubic interpolation)")
    for true in (0.0, 3.5, -6.25), (0.0, -2.75, 1.4):
        moved = ndi.shift(base, [-t for t in true], order=3, mode="wrap")
        got = pcc_bounded(base, moved)
        err = np.abs(np.array(got) - np.array(true))
        good = err.max() < 0.5
        ok &= good
        print(f"  true={true}  recovered={np.round(got, 3)}  "
              f"max err {err.max():.3f}  {'OK' if good else 'MISMATCH'}")

    print("\nbounded search rejects a decoy peak")
    # two superimposed copies at +5 and +45 in y: the global max is ambiguous
    mixed = (0.45 * np.roll(base, -5, axis=1) + 0.55 * np.roll(base, -45, axis=1))
    free = pcc_bounded(base, mixed)
    bound = pcc_bounded(base, mixed, max_shift=(3, 15, 15))
    print(f"  unbounded     -> {np.round(free, 2)}  (takes the stronger, +45)")
    print(f"  bounded(+-15) -> {np.round(bound, 2)}  (must stay within +-15 in y)")
    ok &= abs(bound[1]) <= 15.5 and abs(free[1] - 45) < 1.0

    print("\nagreement with stock skimage on an unambiguous case")
    from skimage.registration import phase_cross_correlation
    moved = ndi.shift(base, [-1, -9, 6], order=3, mode="wrap")
    sk = phase_cross_correlation(base, moved, upsample_factor=10,
                                 normalization="phase")[0]
    mine = pcc_bounded(base, moved)
    print(f"  skimage={np.round(sk, 3)}   ours={np.round(mine, 3)}")
    ok &= np.abs(np.array(sk) - np.array(mine)).max() < 0.2

    print("\nALL CHECKS PASSED" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)
