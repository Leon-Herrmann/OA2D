# OA2D in Abaqus

The off-axis 2D user element `OA2D.f` (from the [DTU off-axis 2D user element library](https://gitlab.windenergy.dtu.dk/lapm/off-axis-2d-user-element-library-for-abaqus)) and the scripts that build the laminate models around it.

## Requirements

Abaqus (2019 or 2021) with a Fortran compiler for user subroutines (Intel Fortran).

## Usage

From this folder, run one configuration:

```bash
abaqus cae noGUI=Config_tunnelcrack.py
```

| script | computes |
|---|---|
| `Config_tunnelcrack.py` | the energy release rate of a tunnel crack |
| `Config_multipletunnelcracks.py` | several tunnel cracks |
| `Config_nonstraightcrack.py` | a crack path through a fibre microstructure (`fibers.txt`, `crack.dat`) |
| `Config_axialstiffness_VCCT.py` | the axial stiffness and the delamination energy release rate (VCCT) |
| `Config_fatigue.py` | crack growth under tension-tension fatigue |

`LaminateModel.py` builds the model and evaluates the results, `utilities.py` holds the material, microstructure and laminate theory helpers, and `SubConfig` is an example Slurm script.

## Tutorials

Step-by-step tutorials of these models on CAE Assistant:

- [Advanced FEA of off-axis tunnel cracking laminates](https://caeassistant.com/product/fea-off-axis-tunnel-cracking-laminates/)
- [An efficient stiffness degradation composites model](https://caeassistant.com/product/stiffness-degradation-model-layered-composites-cracks/)
