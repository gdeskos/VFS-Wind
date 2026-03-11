# 3D Heave Decay Cylinder Test Case

## Description

This test case simulates a cylinder undergoing free heave decay motion in a two-phase fluid (water/air interface). The cylinder is released from an initial displacement and oscillates due to the restoring buoyancy force.

## Physical Parameters

- Reynolds number: 6666.67
- Gravity: -9.81 m/s^2 (Y-direction)
- Density ratio: 1000:1 (water:air)
- Mass ratio: 0.25
- Initial Y position: 0.0254 m

## Missing Files

**IMPORTANT:** This test case requires a grid file (`grid.dat`) that is not included in the repository. You need to generate or provide this file before running.

### Grid Requirements

The grid should be a 3D structured mesh suitable for:
- Two-phase flow simulation with level set
- Immersed boundary method for the cylinder
- Sufficient resolution near the free surface (dthick = 0.006)

## Running the Simulation

Once you have the grid file:

```bash
cd examples/05_3D_Heave_Decay_Cylinder
mpirun -np 8 ../../build/Source/vwis -xml control.xml
```

## Reference Results

The file `_FSI_position00_good_run` contains reference FSI position data from a previous successful run.

## Files

- `control.xml` - Main configuration file (XML format)
- `control.dat` - Legacy configuration file (for reference)
- `bcs.dat` - Boundary conditions
- `ibmdata00` - Immersed boundary mesh data
- `_FSI_position00_good_run` - Reference FSI position output
