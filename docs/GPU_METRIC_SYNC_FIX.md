# GPU Build Grid File Path Fix

## Issue Description

When running VFS-Wind with GPU acceleration enabled (`ENABLE_GPU=ON`) using the XML configuration, simulations would fail at the first timestep with NaN errors:

```
CFL 1.0 time step=0.000000, dx_min=0.00000
*** Max Ucat = nan
ERROR detected by Hypre ... BEGIN
ERROR -- hypre_GMRESSolve: INFs and/or NaNs detected in input.
```

The symptoms included:
- `dx_min = 0` (should be positive mesh spacing)
- `Inlet Area = 0` (should be positive cross-sectional area)
- `Max Ucat = nan` (velocity becomes NaN)
- Hypre solver failures due to NaN values in the RHS vector

## Root Cause

The issue was caused by a **hardcoded grid file path** in `init.c` that ignored the user-specified grid filename when using the XYZ format.

### Technical Details

When the XML configuration specified:
```xml
<grid file="mesh.xyz" format="xyz"/>
```

The XML parser correctly set both options:
- `-grid mesh.xyz` (the grid filename)
- `-xyz 1` (use XYZ format parsing)

However, in `Source/init.c`, the code was:
```c
if(xyz_input) sprintf(str, "%s/%s", path, "xyz.dat");  // HARDCODED!
else sprintf(str, "%s/%s", path, gridfile);
```

This caused the code to look for `xyz.dat` instead of the user-specified `mesh.xyz`, resulting in either:
1. File not found (if `xyz.dat` doesn't exist)
2. Wrong file content (if `xyz.dat` exists but has different content)

When the mesh coordinates weren't loaded correctly, all metric computations produced zeros, leading to:
1. `Calc_Minimum_dt()` computing `dx_min = 0` from zero metric values
2. `Calc_Inlet_Area()` computing `Inlet Area = 0`
3. Division by zero in time step calculations
4. NaN propagation through velocity and pressure fields

### Why GPU builds exposed this issue

The bug existed in both CPU and GPU builds, but was masked in certain configurations where:
- The example directory happened to have both `mesh.xyz` and `xyz.dat` with identical content
- Legacy users without XML configs used the expected `xyz.dat` filename

With the new XML-based configuration and GPU testing, the mismatch became apparent.

## The Fix

### File: `Source/init.c`

Changed the grid file path logic to always use the `gridfile` variable:

**Before:**
```c
if(xyz_input) sprintf(str, "%s/%s", path, "xyz.dat");
else sprintf(str, "%s/%s", path, gridfile);
```

**After:**
```c
/* Use gridfile for all input formats - xyz_input only affects the parsing format */
sprintf(str, "%s/%s", path, gridfile);
```

### File: `Source/main.c`

Added backward compatibility for legacy users who expect `xyz.dat` as the default:

```c
/* Backward compatibility: if xyz_input is set but gridfile is still default, use "xyz.dat" */
if(xyz_input && strcmp(gridfile, "grid.dat") == 0) {
    sprintf(gridfile, "xyz.dat");
}
```

This ensures:
- **XML users**: The specified filename (e.g., `mesh.xyz`) is used
- **Legacy users**: `-xyz 1` without `-grid` still defaults to `xyz.dat`

## Testing

After applying this fix, verify the simulation runs correctly:

```bash
cd examples/02_ChannelFlow
ENABLE_GPU=1 ./run.sh simulate
```

Expected output should show:
- `Reading mesh.xyz` (correct filename)
- `dx_min > 0` (positive mesh spacing)
- `Inlet Area > 0` (positive cross-sectional area)
- `Max Ucat` is a finite positive number
- Simulation completes without Hypre NaN errors

Verified output:
```
Reading mesh.xyz 1.000000e+00, 121x41x61
...
Finish Flow Solver
 ******* Finished computation ti=101 *******
```

## Related Files

- `Source/init.c` - Grid file loading (primary fix)
- `Source/main.c` - Backward compatibility logic
- `Source/xml_input.cpp` - XML parsing that sets `-grid` and `-xyz` options
- `Source/metrics.c` - Metric computation from coordinates
- `Source/bcs.c` - `Calc_Inlet_Area()` that uses metric vectors

## Date

February 2026

## Author

GPU Portability Branch Development
