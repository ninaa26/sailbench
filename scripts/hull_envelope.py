"""Hydrostatics for a shelled CAD hull, by filling its outer envelope.

`hull_analysis.py` measures the volume a mesh encloses. That is the right
question for a solid foil and the wrong one for this boat's hull, which is
modelled as a thin shell: the mesh encloses ~8.8 L of laminate, while floating
27 kg needs 27 L of displacement.

So instead of enclosed volume, this fills the envelope. Vertical rays are cast
on a grid over the waterplane; for each column the first entry and last exit of
the body give the outer skin, and everything between them is inside the hull,
laminate and air alike. That is what actually displaces water.

Usage:
    uv run python scripts/hull_envelope.py boat.stl --mass 27.0 --stations 9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hull_analysis import autoscale, load_stl, normals_and_areas, split_components, volume_and_centroid

Array = NDArray[np.float64]


class Envelope:
    """First-entry / last-exit surfaces of one body, sampled on a waterplane grid."""

    def __init__(self, tris: Array, up: int, fwd: int, lat: int, pitch: float) -> None:
        self.up, self.fwd, self.lat, self.pitch = up, fwd, lat, pitch
        pts = tris.reshape(-1, 3)
        self.lo = pts.min(axis=0)
        self.hi = pts.max(axis=0)

        self.nf = max(1, int(np.ceil((self.hi[fwd] - self.lo[fwd]) / pitch)))
        self.nl = max(1, int(np.ceil((self.hi[lat] - self.lo[lat]) / pitch)))
        self.ymin = np.full((self.nf, self.nl), np.inf)
        self.ymax = np.full((self.nf, self.nl), -np.inf)

        self._rasterise(tris)

    def _rasterise(self, tris: Array) -> None:
        f, l, u, pitch = self.fwd, self.lat, self.up, self.pitch
        # Cell centres, so a ray never lands exactly on a shared edge.
        f0 = self.lo[f] + 0.5 * pitch
        l0 = self.lo[l] + 0.5 * pitch

        for tri in tris:
            af, al, au = tri[0, f], tri[0, l], tri[0, u]
            bf, bl, bu = tri[1, f], tri[1, l], tri[1, u]
            cf, cl, cu = tri[2, f], tri[2, l], tri[2, u]

            det = (bl - cl) * (af - cf) + (cf - bf) * (al - cl)
            if abs(det) < 1e-14:  # edge-on to the ray: contributes no interval
                continue

            i0 = max(0, int(np.floor((min(af, bf, cf) - f0) / pitch)))
            i1 = min(self.nf - 1, int(np.ceil((max(af, bf, cf) - f0) / pitch)))
            j0 = max(0, int(np.floor((min(al, bl, cl) - l0) / pitch)))
            j1 = min(self.nl - 1, int(np.ceil((max(al, bl, cl) - l0) / pitch)))
            if i1 < i0 or j1 < j0:
                continue

            fs = f0 + np.arange(i0, i1 + 1) * pitch
            ls = l0 + np.arange(j0, j1 + 1) * pitch
            ff, ll = np.meshgrid(fs, ls, indexing="ij")

            w0 = ((bl - cl) * (ff - cf) + (cf - bf) * (ll - cl)) / det
            w1 = ((cl - al) * (ff - cf) + (af - cf) * (ll - cl)) / det
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue

            y = w0 * au + w1 * bu + w2 * cu
            block_min = self.ymin[i0 : i1 + 1, j0 : j1 + 1]
            block_max = self.ymax[i0 : i1 + 1, j0 : j1 + 1]
            np.minimum(block_min, np.where(inside, y, np.inf), out=block_min)
            np.maximum(block_max, np.where(inside, y, -np.inf), out=block_max)

    @property
    def filled(self) -> NDArray[np.bool_]:
        """Columns that actually hit the body."""
        return np.isfinite(self.ymin) & np.isfinite(self.ymax)

    def volume_below(self, height: float, *, open_top: bool = False) -> float:
        """Buoyant volume under a horizontal plane.

        Integrates from the outer bottom skin (`ymin`) up to the waterline, and
        deliberately ignores `ymax`. The deck is open over the cockpit, so a ray
        there exits at the inner skin and `ymax` would truncate the column to the
        laminate thickness. What displaces water is everything the outer skin
        encloses below the waterline, whether or not there is deck overhead --
        valid as long as the sheerline stays dry, which `swamped_by` checks.
        """
        ok = self.filled & (self.ymin < height)
        if not ok.any():
            return 0.0
        if open_top:
            return float(np.sum(height - self.ymin[ok]) * self.pitch**2)
        # A closed solid (a foil, a bulb) displaces only its own body, so the
        # column stops at the top of the body rather than at the waterline.
        return float(np.sum(np.minimum(self.ymax[ok], height) - self.ymin[ok]) * self.pitch**2)

    def swamped_by(self, height: float) -> float:
        """How far the waterline sits above the lowest point of the sheer."""
        # Only columns with real depth count as sheer; slivers at the very edge
        # of the footprint have ymax ~= ymin and would always read as swamped.
        real = self.filled & ((self.ymax - self.ymin) > 0.02)
        edge = self.ymax[real]
        return float(height - edge.min()) if len(edge) else 0.0

    def waterplane(self, height: float) -> tuple[float, float, float]:
        """Waterplane area, and the length and beam of the wetted footprint."""
        ok = self.filled & (self.ymin < height)
        if not ok.any():
            return 0.0, 0.0, 0.0
        idx_f, idx_l = np.nonzero(ok)
        area = float(ok.sum()) * self.pitch**2
        length = float(idx_f.max() - idx_f.min() + 1) * self.pitch
        beam = float(idx_l.max() - idx_l.min() + 1) * self.pitch
        return area, length, beam

    def station_positions(self) -> Array:
        return self.lo[self.fwd] + (np.arange(self.nf) + 0.5) * self.pitch

    def section(self, i: int, height: float) -> tuple[float, float, float]:
        """Submerged area, beam at the waterline, and draft at one station."""
        col_min, col_max = self.ymin[i], self.ymax[i]
        ok = np.isfinite(col_min) & (col_min < height)
        if not ok.any():
            return 0.0, 0.0, 0.0
        area = float(np.sum(height - col_min[ok]) * self.pitch)
        j = np.nonzero(ok)[0]
        beam = float(j.max() - j.min() + 1) * self.pitch if len(j) else 0.0
        return area, beam, height - float(col_min[ok].min())


def main() -> None:
    """Fill the hull, float it, and report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stl", type=Path)
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--up", type=int, default=1)
    parser.add_argument("--fwd", type=int, default=0)
    parser.add_argument("--pitch", type=float, default=0.004, help="grid spacing [m]")
    parser.add_argument("--stations", type=int, default=9)
    parser.add_argument("--body", type=int, default=0, help="which body is the hull, by shell volume rank")
    args = parser.parse_args()

    up, fwd = args.up, args.fwd
    lat = ({0, 1, 2} - {up, fwd}).pop()

    tris, note = autoscale(load_stl(args.stl), "auto")
    print(f"{args.stl.name}: {len(tris)} triangles, units {note}")

    bodies = split_components(tris)
    # Identify the hull by enclosed (laminate) volume. Bounding box picks the
    # sail, which is 9 triangles spanning the whole rig; triangle count picks a
    # finely tessellated motor. Shell volume ranks the hull first.
    bodies.sort(key=lambda b: abs(volume_and_centroid(b, np.zeros(3))[0]), reverse=True)

    hull = bodies[args.body]
    hlo = hull.reshape(-1, 3).min(axis=0)
    hhi = hull.reshape(-1, 3).max(axis=0)
    print(
        f"hull body [{args.body}]: {len(hull)} triangles, "
        f"extent {hhi[fwd] - hlo[fwd]:.3f} x {hhi[lat] - hlo[lat]:.3f} x {hhi[up] - hlo[up]:.3f} m"
    )

    env = Envelope(hull, up, fwd, lat, args.pitch)
    print(f"envelope sampled on a {env.nf} x {env.nl} grid at {args.pitch * 1000:.0f} mm pitch")
    print(f"  hull volume to the sheer {env.volume_below(env.hi[up], open_top=True) * 1000:.2f} L")

    # Appendages: any body reaching below the hull's lowest point.
    appendages = [b for b in bodies[1:] if b.reshape(-1, 3)[:, up].min() < env.lo[up] + 1e-6]
    app_envs = [Envelope(b, up, fwd, lat, args.pitch) for b in appendages]

    def displaced(h: float) -> float:
        return env.volume_below(h, open_top=True) + sum(a.volume_below(h) for a in app_envs)

    target = args.mass / args.rho
    lo, hi = env.lo[up] - 1.0, env.hi[up]
    if displaced(hi) < target:
        print(f"\nCANNOT FLOAT: full envelope displaces {displaced(hi) * 1000:.1f} L, needs {target * 1000:.1f} L")
        return
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if displaced(mid) < target:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi)

    wp_area, lwl, bwl = env.waterplane(h)
    hull_disp = env.volume_below(h, open_top=True)
    draft = h - float(env.lo[up])
    block = hull_disp / (lwl * bwl * draft) if lwl * bwl * draft > 0 else float("nan")

    _, areas = normals_and_areas(hull)
    print(f"\nfloats at axis-{up} = {h:.4f} m carrying {args.mass:.1f} kg")
    print(f"  displacement total  {displaced(h) * 1000:8.2f} L")
    print(f"    hull              {hull_disp * 1000:8.2f} L")
    for body, a in zip(appendages, app_envs, strict=True):
        blo = body.reshape(-1, 3).min(axis=0)
        print(f"    appendage at x={blo[fwd]:+.3f}  {a.volume_below(h) * 1000:6.2f} L")
    print(f"  waterline length    {lwl:8.4f} m   <- hull.L")
    print(f"  waterline beam      {bwl:8.4f} m   <- hull.B")
    print(f"  hull draft          {draft:8.4f} m   <- hull.T")
    print(f"  waterplane area     {wp_area:8.4f} m^2")
    print(f"  block coefficient   {block:8.3f}")
    print(f"  hull top above WL   {env.hi[up] - h:8.4f} m")
    margin = env.swamped_by(h)
    verdict = "SWAMPED" if margin > 0 else f"{-margin:.4f} m of freeboard at the lowest point of the sheer"
    print(f"  swamping check      {verdict}")
    print(f"  hull shell area     {float(areas.sum()):8.4f} m^2 (both skins; wetted is a fraction of this)")

    print(f"\nstations (submerged area / beam at WL / draft), {args.stations} along the waterline:")
    xs = env.station_positions()
    inside = np.nonzero(env.filled.any(axis=1))[0]
    picks = np.linspace(inside.min(), inside.max(), args.stations + 2)[1:-1].astype(int)
    print(f"  {'x [m]':>8} {'area m^2':>10} {'beam m':>8} {'draft m':>8}")
    for i in picks:
        area, beam, dr = env.section(int(i), h)
        print(f"  {xs[i]:8.3f} {area:10.5f} {beam:8.4f} {dr:8.4f}")


if __name__ == "__main__":
    main()
