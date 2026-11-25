# run with a kitware pvpython (6.x) after tunnel_crack.py: `pvpython render_tunnel_crack.py`
from pathlib import Path

import numpy as np
from paraview.servermanager import Fetch
from paraview.simple import (
    Box,
    Clip,
    ColorBy,
    CreateView,
    Delete,
    ExtractSurface,
    GetColorTransferFunction,
    HideScalarBarIfNotNeeded,
    LegacyVTKReader,
    LinearExtrusion,
    Render,
    SaveScreenshot,
    Show,
    Slice,
    Threshold,
    WarpByVector,
)
from PIL import Image
from vtkmodules.util.numpy_support import vtk_to_numpy

RESULTS = Path(__file__).parent / "results"

# -------------------------------------- settings -------------------------------------
# colors, of the strain along the load
COLORMAP = (
    "Rainbow Desaturated"  # ParaView's, as cuvox's .cmap/rainbow_desaturated.cmap
)
PERCENTILES = (1.0, 99.0)  # of the off-axis field, the color range of both panels
AMBIENT = 0.5  # share of unshaded light, so the colors stay those of the colormap

# geometry, as in tunnel_crack.py
THETA = 30.0  # degrees, the crack's angle to the load
PLY = 0.1  # mm, the cracked plies from PLY to 3 PLY in height
WIDTH_3D = 0.8  # mm, the 3D block across the load, drawn around the off-axis section
STRAIN = 1e-3  # along the load, which stretches that block as the section
CROP = 0.6  # mm along the load either side of the centre, shown
SCALE = 150.0  # displacement magnification
OPACITY = 0.15  # of the translucent block around what is shown
CRACK = (0.85, 0.1, 0.1, 0.3)  # the crack surface: rgb and opacity

# camera, degrees from a view along -z onto the slice, y up: from the front right, above
AZIMUTH, ELEVATION = 25.0, 15.0

# image
WIDTH = 2800  # of the whole figure, two panels side by side
SUPERSAMPLE = 2  # rendered at this multiple and downscaled, for anti-aliasing
NAME = "tunnel_crack"


# -------------------------------------- pipeline -------------------------------------
def clipped(source, origin, normal, keep_below: bool):
    """The voxels of `source` on one side of the plane through `origin` with `normal`, whole."""
    clip = Clip(Input=source)
    clip.ClipType = "Plane"
    clip.ClipType.Origin = list(origin)
    clip.ClipType.Normal = list(normal)
    clip.Crinkleclip = 1
    clip.Invert = 1 if keep_below else 0
    return clip


def warped(source):
    """`source` displaced by the magnified displacement."""
    warp = WarpByVector(Input=source)
    warp.Vectors = ["POINTS", "displacement"]
    warp.ScaleFactor = SCALE
    return ExtractSurface(Input=warp)


def colored(display, low: float, high: float) -> None:
    """Color `display` by the strain cells from `low` to `high`."""
    display.SetRepresentationType("Surface")
    display.Specular = 0.0
    display.Ambient, display.Diffuse = AMBIENT, 1.0 - AMBIENT
    ColorBy(display, ("CELLS", "strain"))
    lut = GetColorTransferFunction("strain")
    lut.AutomaticRescaleRangeMode = "Never"  # else showing another part widens it
    lut.ApplyPreset(COLORMAP, True)
    lut.RescaleTransferFunction(low, high)


def view(size: tuple[int, int]):
    """A render view of `size` pixels without axes, on a transparent background."""
    v = CreateView("RenderView")
    v.ViewSize = list(size)
    v.OrientationAxesVisibility = 0
    v.UseColorPaletteForBackground = 0
    v.Background = [1.0, 1.0, 1.0]
    return v


def place_camera(v) -> None:
    """The same oblique view in every panel."""
    Render(v)
    camera = v.GetActiveCamera()
    camera.SetViewUp(0.0, 1.0, 0.0)
    focal = camera.GetFocalPoint()
    camera.SetPosition(focal[0], focal[1], focal[2] + 1.0)
    camera.Azimuth(AZIMUTH)
    camera.Elevation(ELEVATION)
    camera.OrthogonalizeViewUp()
    v.ResetCamera(False)


def save(v, path: Path, size: tuple[int, int]) -> None:
    """The view as a transparent PNG at `size`, supersampled."""
    SaveScreenshot(
        str(path),
        v,
        ImageResolution=[SUPERSAMPLE * s for s in size],
        TransparentBackground=1,
    )
    Image.open(path).resize(size, Image.LANCZOS).save(path)


def crack_surface(display) -> None:
    """Draw `display` as the crack: dark and translucent, both faces."""
    display.SetRepresentationType("Surface")
    display.ColorArrayName = [None, ""]
    display.AmbientColor = display.DiffuseColor = list(CRACK[:3])
    display.Opacity = CRACK[3]


def ghost(display) -> None:
    """Draw `display` as what is cut away: grey and translucent."""
    display.SetRepresentationType("Surface")
    display.ColorArrayName = [None, ""]
    display.Opacity = OPACITY
    display.AmbientColor = display.DiffuseColor = [0.6, 0.6, 0.65]


def voxels_of(reader, low: float, high: float):
    """The voxels of `reader` with `solid` in [`low`, `high`] within CROP along x of its centre."""
    voxels = Threshold(Input=reader)
    voxels.Scalars = ["CELLS", "solid"]
    voxels.LowerThreshold, voxels.UpperThreshold = low, high
    b = Fetch(reader).GetBounds()
    centre = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
    for side in (1, -1):
        edge = [centre[0] + side * CROP, centre[1], centre[2]]
        voxels = clipped(voxels, edge, [side, 0.0, 0.0], keep_below=True)
    return voxels


def solid(path: Path):
    """The reader of `path` and its solid voxels within CROP along x of its centre."""
    reader = LegacyVTKReader(FileNames=[str(path)])
    return reader, voxels_of(reader, 0.5, 1.0)


def ghost_box(centre, v) -> None:
    """The cropped 3D block about `centre` in `v`, translucent: stretched as the magnified far field stretches it, a little larger so its faces lie outside the solid's."""
    block = Box()
    block.Center = list(centre)
    block.XLength = 2 * CROP * (1 + SCALE * STRAIN) * 1.002
    block.YLength, block.ZLength = 4 * PLY * 1.002, WIDTH_3D * 1.002
    ghost(Show(block, v))


# the color range from the off-axis field
reader2D, section = solid(RESULTS / "tunnel_crack2D.vtk")
cells = Fetch(reader2D).GetCellData()
field = vtk_to_numpy(cells.GetArray("strain"))[
    vtk_to_numpy(cells.GetArray("solid")) > 0
]
low, high = np.percentile(field, PERCENTILES)
size = (WIDTH // 2, WIDTH // 2)

# --------------------------------------- 3D panel ------------------------------------
# the half behind the mid-depth slice; of it the upper half of the laminate only for
# x below the centre; the block translucent around it, the whole crack (its voids) red
reader3D, block = solid(RESULTS / "tunnel_crack3D.vtk")
b = Fetch(reader3D).GetBounds()
centre = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
back = clipped(block, centre, [0.0, 0.0, 1.0], keep_below=True)
lower = clipped(back, centre, [0.0, 1.0, 0.0], keep_below=True)
top = clipped(back, centre, [0.0, 1.0, 0.0], keep_below=False)
upper = clipped(top, centre, [1.0, 0.0, 0.0], keep_below=True)
v3D = view(size)
for part in (lower, upper):
    colored(Show(warped(part), v3D), low, high)
crack_surface(Show(warped(voxels_of(reader3D, -0.5, 0.5)), v3D))
ghost_box(centre, v3D)
HideScalarBarIfNotNeeded(GetColorTransferFunction("strain"), v3D)
place_camera(v3D)
left = RESULTS / f"{NAME}_3D.png"
save(v3D, left, size)

# --------------------------------------- 2D panel ------------------------------------
# the off-axis section in the same view, as the slice it stands for, with its crack in
# 3D: the deformed slot carried along t = (cos, 0, sin) to the faces of the cropped
# block, which the magnified far field stretches along x
v2D = view(size)
colored(Show(warped(section), v2D), low, high)
b2 = Fetch(reader2D).GetBounds()
angle = np.radians(THETA)
t = np.array([np.cos(angle) * (1 + SCALE * STRAIN), 0.0, np.sin(angle)])
slot = voxels_of(reader2D, -0.5, 0.5)
reach = min(CROP / np.cos(angle), WIDTH_3D / 2 / np.sin(angle))  # along t in the block
oval = Slice(Input=slot)  # the slot's open face mid-slab, which the sweep extrudes
oval.SliceType = "Plane"
oval.SliceType.Origin = [0.0, 0.0, (b2[4] + b2[5]) / 2]
oval.SliceType.Normal = [0.0, 0.0, 1.0]
for direction in (1, -1):  # to the block's faces along t either way
    sweep = LinearExtrusion(Input=warped(oval))
    sweep.Vector = list(direction * t)
    sweep.ScaleFactor = reach
    sweep.Capping = 1
    crack_surface(Show(sweep, v2D))
ghost_box([(b2[0] + b2[1]) / 2, 2 * PLY, (b2[4] + b2[5]) / 2], v2D)
HideScalarBarIfNotNeeded(GetColorTransferFunction("strain"), v2D)
place_camera(v2D)
right = RESULTS / f"{NAME}_2D.png"
save(v2D, right, size)

# -------------------------------------- figure ---------------------------------------
figure = Image.new("RGBA", (WIDTH, WIDTH // 2), (0, 0, 0, 0))
figure.paste(Image.open(left), (0, 0))
figure.paste(Image.open(right), (WIDTH // 2, 0))
figure.save(RESULTS / f"{NAME}.png", optimize=True)
for v in (v3D, v2D):
    Delete(v)
