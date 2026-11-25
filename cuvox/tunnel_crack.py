import itertools
import time
from pathlib import Path
from typing import NamedTuple

import cupy as cp
import numpy as np

from cuvox.core.grid import Grid
from cuvox.core.output import write_vtk
from cuvox.fem.elasticity import Elasticity
from cuvox.fem.elements import centre_gradients
from cuvox.fem.materials.elasticity import isotropic, off_axis, orthotropic
from cuvox.fem.space import Labels, Space

RESULTS = Path(__file__).parent / "results"

# -------------------------------------- settings -------------------------------------
# discretization
PLY_VOXELS_2D = 128  # voxels across one ply of the off-axis model
PLY_VOXELS = 38  # of the 3D model: 223M dofs, 28 GB of a 32 GB GPU (lower it for less)

# geometry, mm: the laminate [0/theta]s loaded along x, a tunnel crack in its theta plies
PLY = 0.1  # ply thickness
THETA = np.radians(30.0)  # angle of the cracked plies' fibres and the crack to the load
CELL = 1.4  # across the crack between the far-field faces, 7 cracked-layer heights
WIDTH = 0.8  # across the load of the 3D model, which the crack crosses
PERIOD = 0.4  # of the fibre arrangement across t
FIBRE_RADIUS = 8.5e-3
FIBRE_FRACTION = 0.5
SEED = 0

# physics, MPa
PLY_MATERIAL = orthotropic(30620, 8620, 8620, 0.33, 0.29, 0.29, 2900, 3250, 3250)
FIBRE = (72000.0, 0.22)  # glass, E and nu
MATRIX = (3500.0, 0.35)  # epoxy
STRAIN = 1e-3  # along the load

# solver
ETA = 1e-6  # relative residual stop
MAXITER = 2000

# output
SHOWN = 1.2  # mm along the load of the 3D model written to its VTK, about the crack
RECORDED = 5.595e-4  # tunnel crack energy release rate of the off-axis model, N/mm

# --------------------------------------- setup ---------------------------------------
# x along the load, y through the thickness, z across; the crack and the cracked plies'
# fibres along t = (cos, 0, sin) of THETA; n = (x, z) . (sin, -cos) the distance across t
s, c = np.sin(THETA), np.cos(THETA)
farfield = STRAIN * np.array([1.0, 0, 0, 0, 0, 0])  # xx, yy, zz, yz, xz, xy


class Discretization(NamedTuple):
    """The voxels of both models at one resolution."""

    h: float  # voxel edge, mm
    section: tuple[int, int]  # voxels of the off-axis model, along the load and up
    margin: int  # voxels the 3D model extends beyond it either side along the load
    box: tuple[int, int, int]  # voxels of the 3D model


def discretization(ply_voxels: int) -> Discretization:
    """The voxels at `ply_voxels` across a ply: the off-axis model CELL across the crack, the 3D model its length plus the crack's run along x."""
    h = PLY / ply_voxels
    section = (round(CELL / s / h), 4 * ply_voxels)
    margin = round(WIDTH * abs(c / s) / (2 * h))
    box = (section[0] + 2 * margin, section[1], 2 * round(WIDTH / (2 * h)))
    return Discretization(h, section, margin, box)


# fibres in the cracked plies: circles across t, random sequential addition in (n, y),
# periodic in n
rng = np.random.default_rng(SEED)
count = round(FIBRE_FRACTION * PERIOD * 2 * PLY / (np.pi * FIBRE_RADIUS**2))
centres = []
for _ in range(200 * count):
    if len(centres) == count:
        break
    p = rng.uniform([0, PLY + FIBRE_RADIUS], [PERIOD, 3 * PLY - FIBRE_RADIUS])
    gaps = [
        np.hypot((p[0] - q[0] + PERIOD / 2) % PERIOD - PERIOD / 2, p[1] - q[1])
        for q in centres
    ]
    if min(gaps, default=np.inf) > 2.1 * FIBRE_RADIUS:
        centres.append(p)
print(f"{len(centres)} fibres")

# the materials of the labels: the 0 plies' fibres along x, fibre, matrix, void
table = np.array([1.0, FIBRE[0], MATRIX[0], 0.0], np.float32)
materials = (
    PLY_MATERIAL,
    isotropic(FIBRE[1]),
    isotropic(MATRIX[1]),
    isotropic(MATRIX[1]),
)


def middle_plies(y: cp.ndarray) -> cp.ndarray:
    """Whether the heights `y` lie in the cracked plies."""
    return (y >= PLY) & (y < 3 * PLY)


def labels_of(n: cp.ndarray, y: cp.ndarray, void: cp.ndarray) -> cp.ndarray:
    """Label 0 the 0 plies, 1 the fibres, 2 the matrix, 3 void, at the points (n, y)."""
    fibre = cp.zeros(void.shape, bool)
    for cn, cy in centres:
        dn = (n - cn + PERIOD / 2) % PERIOD - PERIOD / 2
        fibre |= dn**2 + (y - cy) ** 2 < FIBRE_RADIUS**2
    labels = cp.where(middle_plies(y), cp.where(fibre, 1, 2), 0)
    return cp.where(void, 3, labels).astype(cp.uint8)


def section_labels(d: Discretization, cracked: bool) -> cp.ndarray:
    """The off-axis model's voxels, n across t from the crack in its middle."""
    x = (cp.arange(d.section[0]) + 0.5)[:, None] * d.h
    y = (cp.arange(d.section[1]) + 0.5)[None, :] * d.h
    crack = (cp.arange(d.section[0]) == d.section[0] // 2)[:, None] & middle_plies(y)
    return labels_of((x - d.section[0] * d.h / 2) * s, y, crack & cracked)


def box_labels(d: Discretization, cracked: bool) -> cp.ndarray:
    """The 3D model's voxels about its centre, a laminate block along the grid; the crack the voxels its plane n = 0 crosses (corners on both sides)."""
    x = (cp.arange(d.box[0]) + 0.5)[:, None, None] * d.h - d.box[0] * d.h / 2
    y = (cp.arange(d.box[1]) + 0.5)[None, :, None] * d.h
    z = (cp.arange(d.box[2]) + 0.5)[None, None, :] * d.h - d.box[2] * d.h / 2
    n = x * s - z * c
    spread = d.h / 2 * (abs(s) + abs(c))  # n at a voxel's corners: centre +- spread
    tie = 1e-9 * d.h  # rounding, as cos(pi / 2) is not 0
    crack = (n > tie - spread) & (n <= spread + tie) & middle_plies(y) & cracked
    return labels_of(n, y, crack)


def farfield_displacement(nodes: tuple[int, ...], h: float) -> cp.ndarray:
    """The far-field displacement (3, nodes) of a grid of edge `h`, u = e_xx x about its centre."""
    x = h * (cp.arange(nodes[0], dtype=cp.float32) - (nodes[0] - 1) / 2)
    u = cp.zeros((3, *nodes), cp.float32)
    u[0] += STRAIN * x.reshape(-1, *[1] * (len(nodes) - 1))
    return u


def solve(labels: cp.ndarray, h: float):
    """The strain energy, the displacement u and the model of the voxel `labels` of edge `h`.

    2D: the off-axis model, nothing varying along t, per unit length along z; u the
    fluctuation about the far field, which enters as the load of an eigenstrain. 3D: a
    plain model, u the total displacement, the far field prescribed on the faces x and
    z (u and w).
    """
    ndim = labels.ndim
    space = Space(Grid(labels.shape, h), degree=1, nf=3)
    kind = (lambda m: off_axis(m, THETA)) if ndim == 2 else (lambda m: m)

    # fixed: u and w on the faces x (and z in 3D), v along a line at x = 0
    fixed = cp.zeros(space.shape, bool)
    fixed[[0, 2], 0] = fixed[[0, 2], -1] = True
    fixed[1, 0, labels.shape[1] // 2] = True
    if ndim == 3:
        fixed[[0, 2], :, :, 0] = fixed[[0, 2], :, :, -1] = True
    model = Elasticity(
        space, Labels(labels, table), [kind(m) for m in materials], fixed=fixed
    )

    # 2D: the nodal loads of the far-field strain, -h B^T C e per voxel centre, and the
    # far field's energy; 3D: the far-field displacement on the fixed dofs
    load = cp.zeros(space.shape, cp.float32)
    prescribed, farfield_energy = cp.zeros(space.shape, cp.float32), 0.0
    if ndim == 2:
        B = kind(materials[0]).kinematics(centre_gradients(2, 1))
        nx, ny = labels.shape
        for k, material in enumerate(materials):
            coefficient = (labels == k) * cp.float32(table[k])
            element = -h * B.T @ np.array(material.C) @ farfield
            for a, (i, j) in enumerate(itertools.product((0, 1), repeat=2)):
                for f in range(3):
                    window = (f, slice(i, i + nx), slice(j, j + ny))
                    load[window] += float(element[3 * a + f]) * coefficient
            energy = 0.5 * float(farfield @ np.array(material.C) @ farfield)
            farfield_energy += energy * float(coefficient.sum()) * h**2
    else:
        prescribed = farfield_displacement(space.nodes, h) * fixed
    u, iterations = model.solve(load, prescribed=prescribed, rtol=ETA, maxiter=MAXITER)
    print(f"  {ndim}D, {u.size} dofs, {iterations} iterations")

    # U = u^T A u / 2 - u . f + the far field's energy
    return model.energy(u) - float(cp.vdot(u, load)) + farfield_energy, u, model


def fields(u: cp.ndarray, model: Elasticity, labels: cp.ndarray) -> dict:
    """The strain along the load per voxel, far field included, and the solid voxels."""
    strain = model.gradient(u)[0, 0]  # du/dx, the same in the off-axis model
    if labels.ndim == 2:  # the fluctuation's, plus the far field
        strain += farfield[0]
    solid = labels != 3
    return {"strain": cp.where(solid, strain, 0), "solid": solid}


def write(path: Path, u: cp.ndarray, cells: dict, h: float) -> None:
    """The VTK of the total displacement and the voxel `cells`: of the 3D model within SHOWN along the load about its centre, of the off-axis one (its fluctuation plus the far field along the load) as a slab one voxel thick."""
    if u.ndim == 3:
        u = cp.stack([u, u], -1) + farfield_displacement((*u.shape[1:], 2), h)
        cells = {k: v[..., None] for k, v in cells.items()}
    else:
        first = max(0, u.shape[1] // 2 - round(SHOWN / (2 * h)))
        last = u.shape[1] - first  # nodes first to last - 1, voxels to last - 2
        u = u[:, first:last]
        cells = {k: v[first : last - 1] for k, v in cells.items()}
    write_vtk(
        path,
        (0.0, 0.0, 0.0),
        (h, h, h),
        points={"displacement": u.get()},
        cells={k: cp.asarray(v, cp.float32).get() for k, v in cells.items()},
    )


# --------------------------------------- solve ---------------------------------------
# the off-axis model, intact and cracked; the crack's area per length along z 2 PLY / sin
fine, coarse = discretization(PLY_VOXELS_2D), discretization(PLY_VOXELS)
cp.cuda.Stream.null.synchronize()
tic = time.time()
intact, _, _ = solve(section_labels(fine, False), fine.h)
labels2D = section_labels(fine, True)
cracked, u2D, model2D = solve(labels2D, fine.h)
cp.cuda.Stream.null.synchronize()
print(f"off-axis model {fine.section}: 2 solves in {time.time() - tic:.2f} s")
G = (intact - cracked) / (2 * PLY / s)
print(f"tunnel crack G {G:.4e} N/mm")
RESULTS.mkdir(exist_ok=True)
write(RESULTS / "tunnel_crack2D.vtk", u2D, fields(u2D, model2D, labels2D), fine.h)
del u2D, model2D, labels2D

# the off-axis model at the 3D model's voxels, for the comparison
intact2D, _, _ = solve(section_labels(coarse, False), coarse.h)
cracked2D, u2D, _ = solve(section_labels(coarse, True), coarse.h)
G2D = (intact2D - cracked2D) / (2 * PLY / s)
print(f"at the 3D model's voxels G {G2D:.4e} N/mm")

# the 3D model, intact and cracked; the crack's area 2 PLY times its length WIDTH / sin
cp.cuda.Stream.null.synchronize()
tic = time.time()
intact3D, _, _ = solve(box_labels(coarse, False), coarse.h)
labels3D = box_labels(coarse, True)
cracked3D, u3D, model3D = solve(labels3D, coarse.h)
cp.cuda.Stream.null.synchronize()
print(f"3D model {coarse.box}: 2 solves in {time.time() - tic:.2f} s")
G3D = (intact3D - cracked3D) / (2 * PLY * WIDTH / s)
total2D = u2D + farfield_displacement(u2D.shape[1:], coarse.h)
first = coarse.margin
middle = u3D[:, first : first + coarse.section[0] + 1, :, coarse.box[2] // 2]
gap = cp.abs(middle - total2D).max() / cp.abs(u2D).max()
print(f"3D: G {G3D:.4e} N/mm (the crack's ends at the faces z included), mid-depth")
print(f"displacement off the off-axis one by {float(gap):.2e} of its fluctuation")

# ----------------------------------- postprocessing ----------------------------------
write(RESULTS / "tunnel_crack3D.vtk", u3D, fields(u3D, model3D, labels3D), coarse.h)

assert abs(G / RECORDED - 1) < 1e-2, G
