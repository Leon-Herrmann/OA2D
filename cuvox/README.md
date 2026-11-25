# OA2D in cuvox

The off-axis 2D element on a GPU voxel grid with [cuvox](https://github.com/cmpmech/cuvox), and the 3D model it stands for. `off_axis(material, angle)` gives a material the element's kinematics, and the plies, fibres and matrix are one model: a tuple of materials, one per voxel label, solved by a multigrid-preconditioned CG.

## Installation

An NVIDIA GPU with CUDA 12, then

```bash
pip install cupy-cuda12x
git clone https://github.com/cmpmech/cuvox
pip install -e cuvox
```

and for the figure ParaView 6 (its `pvpython`).

## Usage

```bash
python tunnel_crack.py
pvpython render_tunnel_crack.py
```

`tunnel_crack.py` builds a [0/θ]s glass/epoxy laminate loaded along x, θ = 30° by default, with random fibres in the θ plies and a tunnel crack along them, and computes the energy release rate of the crack from the strain energies of the intact and the cracked laminate:

- the off-axis model: the section of the laminate along the load, nothing varying along the crack, the fibres as ellipses; the far field enters as the load of an eigenstrain and is held on the section's ends, 7 cracked-layer heights apart across the crack;
- the 3D model: a block of the same laminate on an ordinary voxel grid, the crack and the fibre cylinders at θ across it, the far-field displacement prescribed on its faces; its section at mid-depth is the off-axis model's.

It writes `results/tunnel_crack2D.vtk` and `results/tunnel_crack3D.vtk` (the 3D model within 1.2 mm of the crack along the load), which `render_tunnel_crack.py` draws into `results/tunnel_crack.png`. The settings are at the top of each file.

| model | voxels | degrees of freedom | G (N/mm) |
|---|---|---|---|
| off-axis, 128 voxels per ply | 3584 × 512 | 5.5M | 5.595e-4 |
| off-axis, 50 voxels per ply | 1400 × 200 | 0.84M | 6.156e-4 |
| 3D, 50 voxels per ply | 2092 × 200 × 400 | 506M | 6.383e-4 |

The 3D model exceeds the off-axis one at its resolution by 3.7 % and its section at mid-depth differs from the off-axis model's by 15 % of the displacement: the crack ends on the gripped faces of the block, and the voxels resolve the inclined fibres more coarsely than the off-axis model's ellipses (the difference in G was 13 % at 38 voxels per ply, 20 % at 32). The 3D model at its default needs about 27 GB, a 32 GB GPU; `PLY_VOXELS` lowers it (40: 259M degrees of freedom, about 14 GB). On an RTX 5090 its two solves take 14 s, the off-axis model's 0.4 s.
