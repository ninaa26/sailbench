"""Hydrostatics and foil geometry straight off an STL of the boat.

Solves the floating waterline by bisecting submerged volume against the boat's
mass, then reports everything `configs/*.yaml` needs that the CAD mass
properties dialog cannot give: waterline length and beam, draft, displacement,
wetted surface, block coefficient, LCB, hull sections at N stations, and the
span and planform area of each appendage.

Usage:
    uv run python scripts/hull_analysis.py boat.stl --mass 27.0
    uv run python scripts/hull_analysis.py boat.stl --mass 27.0 --stations 11 --yaml out.yaml

Frame: the STL comes out of SolidWorks in the assembly frame, which for this
boat is X longitudinal (+X aft), Y vertical (+Y up), Z transverse. Override with
--up/--fwd if a future export differs. Lengths are auto-detected as mm or m.

The waterline solve assumes the boat floats level at its design attitude. It
does not solve for trim or heel — a real free-floating solution would also
balance pitch, which needs the mass distribution, not just the total.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

EPS = 1e-9
WATER_DENSITY = 1000.0


# ----------------------------------------------------------------------------
# STL loading
# ----------------------------------------------------------------------------


def load_stl(path: Path) -> Array:
    """Load an STL (binary or ASCII) as an (F, 3, 3) array of triangle vertices."""
    raw = path.read_bytes()

    # An ASCII STL starts with "solid", but so do some binary ones written by
    # sloppy exporters, so confirm using the declared triangle count.
    if raw[:5].lower() == b"solid" and not _looks_binary(raw):
        return _load_ascii(raw)
    return _load_binary(raw)


def _looks_binary(raw: bytes) -> bool:
    if len(raw) < 84:
        return False
    (count,) = struct.unpack("<I", raw[80:84])
    return len(raw) == 84 + count * 50


def _load_binary(raw: bytes) -> Array:
    (count,) = struct.unpack("<I", raw[80:84])
    body = np.frombuffer(raw, dtype=np.uint8, count=count * 50, offset=84).reshape(count, 50)
    # Each facet: 12 floats (normal + 3 vertices) then a 2-byte attribute.
    floats = body[:, :48].copy().view(np.float32).reshape(count, 12)
    return floats[:, 3:].reshape(count, 3, 3).astype(np.float64)


def _load_ascii(raw: bytes) -> Array:
    verts: list[list[float]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts and parts[0] == "vertex":
            verts.append([float(v) for v in parts[1:4]])
    if len(verts) % 3:
        msg = f"ASCII STL has {len(verts)} vertices, not a multiple of 3"
        raise ValueError(msg)
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def autoscale(tris: Array, units: str) -> tuple[Array, str]:
    """Convert to metres. 'auto' calls anything bigger than 10 units millimetres."""
    if units == "m":
        return tris, "m (given)"
    if units == "mm":
        return tris / 1000.0, "mm (given)"
    span = float(np.max(np.ptp(tris.reshape(-1, 3), axis=0)))
    if span > 10.0:
        return tris / 1000.0, f"mm (detected, extent {span:.0f})"
    return tris, f"m (detected, extent {span:.2f})"


# ----------------------------------------------------------------------------
# Mesh primitives
# ----------------------------------------------------------------------------


def normals_and_areas(tris: Array) -> tuple[Array, Array]:
    """Return unit normals and areas for each triangle."""
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    areas = 0.5 * norm
    unit = np.divide(cross, np.where(norm[:, None] < EPS, 1.0, norm[:, None]))
    return unit, areas


def volume_and_centroid(tris: Array, apex: Array) -> tuple[float, Array]:
    """Signed volume and centroid of the solid, decomposed into tetrahedra at `apex`.

    Triangles lying in a plane through `apex` contribute nothing, so an open mesh
    that was cut by such a plane can be measured without closing the cut first.
    """
    a = tris[:, 0] - apex
    b = tris[:, 1] - apex
    c = tris[:, 2] - apex
    vols = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    total = float(vols.sum())
    if abs(total) < EPS:
        return 0.0, apex.copy()
    centroids = (apex + tris.sum(axis=1)) / 4.0
    return total, (vols[:, None] * centroids).sum(axis=0) / total


def clip_below(tris: Array, axis: int, height: float) -> tuple[Array, Array]:
    """Keep the part of the mesh with coord[axis] <= height.

    Returns the clipped triangles and the oriented cut segments, shape (S, 2, 3),
    wound consistently with the surface so their loops can be measured directly.
    """
    kept: list[Array] = []
    cuts: list[Array] = []

    dist = tris[:, :, axis] - height
    inside = dist <= 0.0
    n_inside = inside.sum(axis=1)

    fully = tris[n_inside == 3]
    if len(fully):
        kept.append(fully)

    for tri, ins in zip(tris[(n_inside == 1) | (n_inside == 2)], inside[(n_inside == 1) | (n_inside == 2)], strict=True):
        poly, is_exit, is_entry = _clip_polygon(tri, ins, axis, height)
        if len(poly) < 3:
            continue
        pts = np.asarray(poly)
        # Fan triangulation preserves winding.
        for i in range(1, len(pts) - 1):
            kept.append(np.asarray([[pts[0], pts[i], pts[i + 1]]]))
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            if is_exit[i] and is_entry[j]:
                cuts.append(np.asarray([[pts[i], pts[j]]]))

    clipped = np.concatenate(kept) if kept else np.zeros((0, 3, 3))
    segments = np.concatenate(cuts) if cuts else np.zeros((0, 2, 3))
    return clipped, segments


def _clip_polygon(
    tri: Array,
    inside: NDArray[np.bool_],
    axis: int,
    height: float,
) -> tuple[list[Array], list[bool], list[bool]]:
    """Sutherland-Hodgman clip of one triangle, tracking which points are new."""
    poly: list[Array] = []
    is_exit: list[bool] = []
    is_entry: list[bool] = []

    for i in range(3):
        cur, nxt = tri[i], tri[(i + 1) % 3]
        cur_in, nxt_in = bool(inside[i]), bool(inside[(i + 1) % 3])
        if nxt_in:
            if not cur_in:
                poly.append(_intersect(cur, nxt, axis, height))
                is_exit.append(False)
                is_entry.append(True)
            poly.append(nxt)
            is_exit.append(False)
            is_entry.append(False)
        elif cur_in:
            poly.append(_intersect(cur, nxt, axis, height))
            is_exit.append(True)
            is_entry.append(False)

    return poly, is_exit, is_entry


def _intersect(p: Array, q: Array, axis: int, height: float) -> Array:
    denom = q[axis] - p[axis]
    t = 0.0 if abs(denom) < EPS else (height - p[axis]) / denom
    return p + np.clip(t, 0.0, 1.0) * (q - p)


def loop_area(segments: Array, u: int, v: int, apex_u: float, apex_v: float) -> float:
    """Area enclosed by oriented segments, projected onto axes (u, v).

    Shoelace over oriented edges needs no vertex ordering, and edges through the
    apex contribute nothing — which is how open cuts are handled.
    """
    if not len(segments):
        return 0.0
    p = segments[:, 0]
    q = segments[:, 1]
    cross = (p[:, u] - apex_u) * (q[:, v] - apex_v) - (q[:, u] - apex_u) * (p[:, v] - apex_v)
    return abs(float(cross.sum()) / 2.0)


def cross_section(tris: Array, axis: int, position: float) -> Array:
    """Oriented segments where the mesh crosses the plane coord[axis] = position."""
    dist = tris[:, :, axis] - position
    inside = dist <= 0.0
    n_inside = inside.sum(axis=1)
    mask = (n_inside == 1) | (n_inside == 2)
    if not mask.any():
        return np.zeros((0, 2, 3))

    _, segments = clip_below(tris[mask], axis, position)
    return segments


# ----------------------------------------------------------------------------
# Connected components (one per solid body, so per part in the assembly)
# ----------------------------------------------------------------------------


def split_components(tris: Array, tol: float = 1e-6) -> list[Array]:
    """Group triangles into connected bodies by welding near-identical vertices."""
    flat = tris.reshape(-1, 3)
    keys = np.round(flat / tol).astype(np.int64)
    _, vertex_id = np.unique(keys, axis=0, return_inverse=True)
    vertex_id = vertex_id.reshape(-1, 3)

    parent = np.arange(vertex_id.max() + 1)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for tri in vertex_id:
        a = find(int(tri[0]))
        for other in tri[1:]:
            b = find(int(other))
            if a != b:
                parent[b] = a

    roots = np.array([find(int(t[0])) for t in vertex_id])
    return [tris[roots == r] for r in np.unique(roots)]


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------


@dataclass
class Body:
    """One connected solid from the STL."""

    tris: Array
    up: int
    fwd: int
    lat: int
    label: str = "?"
    volume: float = 0.0
    area: float = 0.0
    bbox_min: Array = field(default_factory=lambda: np.zeros(3))
    bbox_max: Array = field(default_factory=lambda: np.zeros(3))


def solve_waterline(tris: Array, up: int, mass: float, rho: float) -> tuple[float, float]:
    """Bisect the waterline height so displaced mass equals the boat's mass."""
    target = mass / rho
    lo = float(tris[:, :, up].min())
    hi = float(tris[:, :, up].max())

    if displaced_volume(tris, up, hi) < target:
        msg = (
            f"the whole hull displaces only {displaced_volume(tris, up, hi) * 1000:.1f} L "
            f"but {target * 1000:.1f} L is needed to float {mass:.1f} kg — it sinks"
        )
        raise ValueError(msg)

    for _ in range(80):
        if hi - lo < 1e-6:
            break
        mid = 0.5 * (lo + hi)
        if displaced_volume(tris, up, mid) < target:
            lo = mid
        else:
            hi = mid

    height = 0.5 * (lo + hi)
    return height, displaced_volume(tris, up, height)


def displaced_volume(tris: Array, up: int, height: float) -> float:
    below, _ = clip_below(tris, up, height)
    if not len(below):
        return 0.0
    apex = np.zeros(3)
    apex[up] = height
    return abs(volume_and_centroid(below, apex)[0])


def planform_area(tris: Array, thin_axis: int) -> float:
    """Silhouette area of a thin body, projected along its thinnest dimension."""
    unit, areas = normals_and_areas(tris)
    return 0.5 * float((areas * np.abs(unit[:, thin_axis])).sum())


def describe_bodies(tris: Array, up: int, fwd: int, lat: int, height: float) -> list[Body]:
    """Measure each connected body and guess what it is."""
    bodies: list[Body] = []
    for part in split_components(tris):
        apex = np.zeros(3)
        apex[up] = height
        _, areas = normals_and_areas(part)
        body = Body(
            tris=part,
            up=up,
            fwd=fwd,
            lat=lat,
            volume=abs(volume_and_centroid(part, apex)[0]),
            area=float(areas.sum()),
            bbox_min=part.reshape(-1, 3).min(axis=0),
            bbox_max=part.reshape(-1, 3).max(axis=0),
        )
        bodies.append(body)

    bodies.sort(key=lambda b: b.volume, reverse=True)
    if bodies:
        bodies[0].label = "hull (largest body)"
    submerged = [b for b in bodies[1:] if b.bbox_min[up] < height]
    submerged.sort(key=lambda b: b.bbox_min[up])
    for i, body in enumerate(submerged[:2]):
        # +fwd is aft in this assembly, so the appendage at larger fwd coord is the rudder.
        body.label = "appendage (deepest)" if i == 0 else "appendage"
    if len(submerged) >= 2:
        by_station = sorted(submerged[:2], key=lambda b: b.bbox_max[fwd])
        by_station[0].label = "keel + bulb (forward appendage)"
        by_station[1].label = "rudder (aft appendage)"
    return bodies


def main() -> None:
    """Run the analysis and print a report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stl", type=Path, help="STL of the full assembly")
    parser.add_argument("--mass", type=float, required=True, help="boat mass [kg] the hull must float")
    parser.add_argument("--rho", type=float, default=WATER_DENSITY, help="water density [kg/m^3]")
    parser.add_argument("--up", type=int, default=1, help="vertical axis index (SolidWorks default: 1 = Y)")
    parser.add_argument("--fwd", type=int, default=0, help="longitudinal axis index (default: 0 = X)")
    parser.add_argument("--stations", type=int, default=11, help="number of hull sections to report")
    parser.add_argument("--units", choices=["auto", "mm", "m"], default="auto")
    parser.add_argument("--yaml", type=Path, default=None, help="also write the numbers here")
    args = parser.parse_args()

    up, fwd = args.up, args.fwd
    lat = ({0, 1, 2} - {up, fwd}).pop()

    tris, unit_note = autoscale(load_stl(args.stl), args.units)
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)

    print(f"{args.stl.name}: {len(tris)} triangles, units {unit_note}")
    print(f"  overall extents  L {hi[fwd] - lo[fwd]:.3f} m   B {hi[lat] - lo[lat]:.3f} m   H {hi[up] - lo[up]:.3f} m")

    height, displaced = solve_waterline(tris, up, args.mass, args.rho)
    below, waterline = clip_below(tris, up, height)
    _, sub_centroid = volume_and_centroid(below, _apex(up, height))
    _, wetted_areas = normals_and_areas(below)

    bodies = describe_bodies(tris, up, fwd, lat, height)
    hull_body = bodies[0] if bodies else None

    lwl = float(waterline[:, :, fwd].max() - waterline[:, :, fwd].min()) if len(waterline) else 0.0
    bwl = float(waterline[:, :, lat].max() - waterline[:, :, lat].min()) if len(waterline) else 0.0
    deep_draft = height - float(lo[up])
    wp_area = loop_area(waterline, fwd, lat, 0.0, 0.0)

    # hull.T in the sim is the canoe body's draft, not the depth of the keel tip,
    # and the block coefficient has to use the same hull-only numbers.
    if hull_body is not None:
        hull_draft = height - float(hull_body.bbox_min[up])
        hull_disp = displaced_volume(hull_body.tris, up, height)
    else:
        hull_draft, hull_disp = deep_draft, displaced
    denom = lwl * bwl * hull_draft
    block = hull_disp / denom if denom > 0 else float("nan")

    print(f"\nfloating waterline at {up}={height:.4f} m (bisected to balance {args.mass:.2f} kg)")
    print(f"  displacement      {displaced * 1000:.2f} L  ({displaced * args.rho:.2f} kg)")
    print(f"    of which hull   {hull_disp * 1000:.2f} L")
    print(f"  waterline length  {lwl:.4f} m   <- hull.L")
    print(f"  waterline beam    {bwl:.4f} m   <- hull.B")
    print(f"  hull draft        {hull_draft:.4f} m   <- hull.T (canoe body only)")
    print(f"  draft to keel tip {deep_draft:.4f} m")
    print(f"  wetted surface    {float(wetted_areas.sum()):.4f} m^2")
    print(f"  waterplane area   {wp_area:.4f} m^2")
    print(f"  block coefficient {block:.3f}  (hull volume / L*B*T)")
    print(f"  centre of buoyancy ({sub_centroid[0]:.4f}, {sub_centroid[1]:.4f}, {sub_centroid[2]:.4f}) m")

    print(f"\nhull sections ({args.stations} stations, from {lo[fwd]:.3f} to {hi[fwd]:.3f} m along axis {fwd}):")
    print(f"  {'station':>10}  {'beam@WL':>9}  {'draft':>8}  {'area':>9}")
    stations = np.linspace(lo[fwd], hi[fwd], args.stations + 2)[1:-1]
    sections = []
    for x in stations:
        seg = cross_section(below, fwd, float(x))
        if not len(seg):
            continue
        pts = seg.reshape(-1, 3)
        at_wl = pts[np.abs(pts[:, up] - height) < 1e-4]
        beam = float(at_wl[:, lat].max() - at_wl[:, lat].min()) if len(at_wl) > 1 else 0.0
        local_draft = height - float(pts[:, up].min())
        area = loop_area(seg, up, lat, height, 0.0)
        sections.append({"x": float(x), "beam": beam, "draft": local_draft, "area": area})
        print(f"  {x:10.3f}  {beam:9.4f}  {local_draft:8.4f}  {area:9.5f}")

    print("\nbodies (connected solids in the STL):")
    appendages = []
    for body in bodies[:8]:
        size = body.bbox_max - body.bbox_min
        thin = int(np.argmin(size))
        span = max(0.0, height - float(body.bbox_min[up]))
        plan = planform_area(body.tris, thin)
        sub, _ = clip_below(body.tris, up, height)
        sub_plan = planform_area(sub, thin) if len(sub) else 0.0
        print(f"  {body.label}")
        print(
            f"      volume {body.volume * 1000:8.2f} L   bbox {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m"
            f"   submerged span {span:.4f} m"
        )
        print(f"      planform {plan:.5f} m^2 (submerged {sub_plan:.5f} m^2)   surface {body.area:.4f} m^2")
        if "keel" in body.label or "rudder" in body.label:
            appendages.append(
                {"label": body.label, "span": span, "planform": plan, "submerged_planform": sub_plan},
            )

    if args.yaml:
        _write_yaml(args, height, displaced, lwl, bwl, hull_draft, float(wetted_areas.sum()), wp_area, block,
                    sections, appendages)
        print(f"\nwrote {args.yaml}")


def _apex(up: int, height: float) -> Array:
    apex = np.zeros(3)
    apex[up] = height
    return apex


def _write_yaml(  # noqa: PLR0913
    args: argparse.Namespace,
    height: float,
    displaced: float,
    lwl: float,
    bwl: float,
    draft: float,
    wetted: float,
    wp_area: float,
    block: float,
    sections: list[dict[str, float]],
    appendages: list[dict[str, float | str]],
) -> None:
    import yaml  # noqa: PLC0415

    payload = {
        "source": {"stl": str(args.stl), "mass_kg": args.mass, "rho": args.rho},
        "waterline": {
            "height_m": round(height, 5),
            "displacement_m3": round(displaced, 6),
            "L_m": round(lwl, 4),
            "B_m": round(bwl, 4),
            "T_m": round(draft, 4),
            "wetted_area_m2": round(wetted, 4),
            "waterplane_area_m2": round(wp_area, 4),
            "block_coefficient": round(block, 4),
        },
        "sections": [{k: round(v, 5) for k, v in s.items()} for s in sections],
        "appendages": appendages,
    }
    args.yaml.write_text(yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
