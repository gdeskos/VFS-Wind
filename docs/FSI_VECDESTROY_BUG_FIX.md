# FSI VecDestroy Bug Fix

## Issue Summary

A SEGV (segmentation fault) crash occurred during VIV (Vortex-Induced Vibration) simulations when using the FSI (Fluid-Structure Interaction) code path with IBM (Immersed Boundary Method).

## Symptoms

- Crash occurred after "CFL 1.0 time step=..." output
- Stack trace showed crash in `Contra2Cart_2` at `DMDAVecGetArray(fda, Coor, &coor)`
- The crash happened on the 4th call to `Contra2Cart_2`, after `Calc_forces_SI` had run
- PETSc error: "Caught signal number 11 SEGV"

## Root Cause

The functions `Calc_forces_SI` and `Calc_forces_SI_levelset` in `Source/fsi.c` incorrectly called `VecDestroy(&Coor)` on the coordinate vector obtained via `DMGetCoordinatesLocal()`.

**The Problem:**
```c
// In Calc_forces_SI (around line 5318)
DMGetCoordinatesLocal(da, &Coor);
DMDAVecGetArray(fda, Coor, &coor);
// ... use coor ...
DMDAVecRestoreArray(fda, Coor, &coor);
VecDestroy(&Coor);  // BUG: This destroys the DM's internal coordinate vector!
```

The coordinate Vec returned by `DMGetCoordinatesLocal()` is **owned by the DM** (Distributed Mesh) object. It is an internal vector managed by PETSc and must NOT be destroyed by user code. Destroying it corrupts the DM's internal state, causing subsequent coordinate access to crash.

## Files Modified

### 1. Source/fsi.c

**Line ~6427 (in `Calc_forces_SI`):**
```c
// Before:
VecDestroy(&Coor);

// After:
// VecDestroy(&Coor);  // DO NOT destroy - Coor is owned by the DM
```

**Line ~7472 (in `Calc_forces_SI_levelset`):**
```c
// Before:
VecDestroy(&Coor);

// After:
// VecDestroy(&Coor);  // DO NOT destroy - Coor is owned by the DM
```

### 2. Source/ibm.c

**Lines ~3673-3700 (in `ibm_interpolation_advanced`):**

Added restore/re-get blocks around the `Contra2Cart()` call to prevent double `DMDAVecGetArray` issues. This was a secondary fix to handle potential array locking conflicts:

```c
// Restore arrays before calling Contra2Cart to avoid double DMDAVecGetArray
DMDAVecRestoreArray(fda, user->lCsi, &csi);
DMDAVecRestoreArray(fda, user->lEta, &eta);
DMDAVecRestoreArray(fda, user->lZet, &zet);
DMDAVecRestoreArray(fda, user->lICsi, &icsi);
DMDAVecRestoreArray(fda, user->lJEta, &jeta);
DMDAVecRestoreArray(fda, user->lKZet, &kzet);
DMDAVecRestoreArray(da, user->lAj, &aj);
DMDAVecRestoreArray(da, user->lNvert, &nvert);
DMDAVecRestoreArray(da, user->lUstar, &ustar);
DMDAVecRestoreArray(fda, user->Ucat, &ucat);

Contra2Cart(user);

// Re-get arrays after Contra2Cart returns
DMDAVecGetArray(fda, user->lCsi, &csi);
DMDAVecGetArray(fda, user->lEta, &eta);
DMDAVecGetArray(fda, user->lZet, &zet);
DMDAVecGetArray(fda, user->lICsi, &icsi);
DMDAVecGetArray(fda, user->lJEta, &jeta);
DMDAVecGetArray(fda, user->lKZet, &kzet);
DMDAVecGetArray(da, user->lAj, &aj);
DMDAVecGetArray(da, user->lNvert, &nvert);
DMDAVecGetArray(da, user->lUstar, &ustar);
DMDAVecGetArray(fda, user->Ucat, &ucat);
```

## PETSc Best Practices

When working with PETSc coordinate vectors:

1. **DO NOT** call `VecDestroy()` on vectors returned by `DMGetCoordinatesLocal()` or `DMGetCoordinates()`
2. These vectors are managed internally by the DM object
3. Only call `DMDAVecRestoreArray()` after `DMDAVecGetArray()` - no destruction needed
4. If you need a copy of coordinates that you own, use `VecDuplicate()` and `VecCopy()`

## Testing

The fix was verified with the `03_VIV_Mounted_Cylinder` example:
```bash
cd examples/03_VIV_Mounted_Cylinder
../../build/Source/vwis -xml control.xml
```

The simulation now runs without crashing through multiple timesteps.

## Related Code Patterns

Other locations in the codebase correctly avoid this bug by commenting out the VecDestroy:
- `Source/bcs.c:738` - `//  VecDestroy(&Coor);`
- `Source/fsi.c:4583` - `//VecDestroy(&Coor);`
- `Source/fsi.c:5123` - `//VecDestroy(&Coor);`
- `Source/ibm.c:2254` - `//  VecDestroy(&Coor);`
- `Source/ibm.c:2845` - `//  VecDestroy(&Coor);`

## Date

March 2026
