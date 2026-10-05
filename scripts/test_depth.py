"""Is the raycaster's near/far convention right?

Two independent checks on synthetic data:

1. OCCLUSION -- two opaque blobs at the same XY, one at the near end of z and
   one at the far end.  With strong absorption only the NEAR one should show.
2. CUTAWAY -- which quadrant actually gets removed, as a function of azimuth.

If the cut lands on the far side, the silhouette stays whole; if it lands on the
near side, the outline has a bite out of it and you can see into the cavity.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import nd2_perfusion_3d as P

xp, ndi, _ = P.backend(True)
nz, ny, nx = 20, 256, 256
dz, dy, dx = 20.266, 2.197, 2.197

# ---------- 1. occlusion ----------
v = np.zeros((nz, ny, nx), np.float32)
yy, xx = np.mgrid[0:ny, 0:nx]
disc = ((yy - ny // 2) ** 2 + (xx - nx // 2) ** 2) < 45 ** 2
v[nz - 1][disc] = 0.35       # DIM blob at the high-z end
v[nz - 2][disc] = 0.35
v[0][disc] = 1.0             # BRIGHT blob at the low-z end
v[1][disc] = 1.0
vol = xp.asarray(v)

rc = P.Raycaster((nz, ny, nx), (dz, dy, dx), 300, 300, xp, ndi)
lut = xp.asarray(P.make_lut("inferno"))
img = rc.render([vol], [lut], [0.0], [1.0], [40.0], 0.0, 0.0,
                n_samples=400, zoom=1.6, emit=[6.0])
h = P.to_host(img)
centre = h[140:160, 140:160].mean()
print("OCCLUSION TEST (az=0, el=0, strong absorption)")
print(f"  dim blob at z index {nz-1}, bright blob at z index 0")
print(f"  rendered centre brightness = {centre:.3f}")
print("  the DIM blob sits at high z, the BRIGHT blob at low z")
print("  -> near end is HIGH z: the dim blob occludes the bright one"
      if centre < 0.55 else
      "  -> near end is LOW z: the bright blob occludes the dim one")

R = rc._rot(0.0, 0.0)
print(f"  R[:,2] (assumed 'toward camera') = {np.round(R[:, 2], 3)}")

# ---------- 2. cutaway vs azimuth ----------
v2 = np.zeros((nz, ny, nx), np.float32)
v2[:, 40:216, 40:216] = 1.0                      # solid slab
# a bright marker bar on the +X side so we can tell orientation
v2[:, 118:138, 190:214] = 2.0
vol2 = xp.asarray(v2)

azs = [0, 45, 90, 135, 180, 225, 270, 315]
fig, axes = plt.subplots(2, len(azs), figsize=(2.3 * len(azs), 5.0))
for j, az in enumerate(azs):
    for row, cut in enumerate((False, True)):
        img = rc.render([vol2], [lut], [0.0], [2.0], [0.9], az, 12.0,
                        n_samples=400, zoom=1.5, emit=[1.6],
                        cutaway=cut, cut_center=(0.0, 0.0))
        ax = axes[row, j]
        ax.imshow(P.to_host(img))
        ax.axis("off")
        if row == 0:
            ax.set_title(f"az {az}", fontsize=9)
axes[0, 0].set_ylabel("no cut")
axes[1, 0].set_ylabel("cut")
plt.suptitle("Cutaway quadrant vs azimuth (el 12). The bite must face the "
             "viewer at every azimuth.", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, .94])
plt.savefig("out2/test_depth.png", dpi=110)
print("\nwrote out2/test_depth.png")
