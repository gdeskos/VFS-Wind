# VFS-Wind Data Structures Reference

This document describes the primary data structures used in VFS-Wind and how
data flows between them.

---

## 1. Main Simulation Context: `UserCtx`

**Defined in:** `Source/variables.h` (lines 249-532)

`UserCtx` is the central data structure that holds all simulation state. Every
major function receives a pointer to this structure.

### Grid and Domain Decomposition

```c
typedef struct UserCtx {
    // PETSc Distributed Arrays (domain decomposition)
    DM da;              // Scalar field layout (pressure, etc.)
    DM fda;             // 3-component vector layout (velocity)
    DM fda2;            // 2-component vector layout

    // Grid dimensions
    PetscInt IM, JM, KM;        // Global grid size
    PetscInt xs, xe, ys, ye;    // Local subdomain range
    PetscInt zs, ze;
    PetscInt lxs, lxe, lys, lye; // Extended range with ghosts
    PetscInt lzs, lze;
```

**Data Flow:**
```
grid.dat/xyz.dat  →  DMDACreate3d()  →  da, fda  →  All vector operations
```

### Velocity and Pressure Fields

```c
    // Primary solution variables
    Vec Ucont;          // Contravariant velocity (flux form)
    Vec Ucat;           // Cartesian velocity (u, v, w)
    Vec P;              // Pressure
    Vec Phi;            // Velocity potential (for corrections)

    // Previous timestep values (for time integration)
    Vec Ucont_o;        // Contravariant velocity at t^n
    Vec Ucat_o;         // Cartesian velocity at t^n
    Vec P_o;            // Pressure at t^n

    // Intermediate values
    Vec Ucont_rm1, Ucont_rm2;   // Sub-stage values for RK
```

**Data Flow:**
```
                    ┌─────────────┐
                    │  Ucat_o     │  (previous timestep)
                    └──────┬──────┘
                           │
                           ▼
┌─────────────┐    ┌───────────────┐    ┌─────────────┐
│ ComputeRHS  │ →  │ Time Advance  │ →  │ Ucont_star  │  (intermediate)
│ (rhs.c)     │    │ (momentum.c)  │    │             │
└─────────────┘    └───────────────┘    └──────┬──────┘
                                               │
                    ┌─────────────┐            │
                    │   Poisson   │ ←──────────┘
                    │  (poisson.c)│
                    └──────┬──────┘
                           │
                           ▼
┌─────────────┐    ┌───────────────┐    ┌─────────────┐
│  Pressure P │ ←  │  Projection   │ →  │ Ucont (new) │
│             │    │               │    │ Ucat (new)  │
└─────────────┘    └───────────────┘    └─────────────┘
```

### Grid Metrics

```c
    // Cell center coordinates
    Vec Cent;           // (x, y, z) of cell centers

    // Contravariant basis vectors (∂x/∂ξ, etc.)
    Vec Csi, Eta, Zet;  // Face normal directions
    Vec Aj;             // Jacobian of transformation

    // Local copies with ghost points
    Vec lCsi, lEta, lZet, lAj;
    Vec lCent;
```

**Data Flow:**
```
xyz.dat  →  FormMetrics()  →  Csi, Eta, Zet, Aj  →  All spatial derivatives
            (metrics.c)         │
                                ▼
                         Stored for reuse throughout simulation
```

### Turbulence Variables

```c
    // LES
    Vec lCs;            // Smagorinsky constant (local)
    Vec lNu_t;          // Eddy viscosity

    // RANS k-omega SST
    Vec K_Omega;        // Combined (k, omega) vector
    Vec lK_Omega;       // Local with ghosts
    Vec lF1;            // SST blending function F1
```

**Data Flow (LES):**
```
Ucat  →  Compute_Smagorinsky_Constant_1()  →  Cs  →  Compute_eddy_viscosity_LES()  →  Nu_t
            (les.c)                                       (les.c)
                                                              │
                                                              ▼
                                                    Used in momentum RHS
```

**Data Flow (RANS):**
```
Ucat  →  Solve_K_Omega()  →  K_Omega  →  Compute Nu_t from k, omega
             (k-omega.c)
```

### Boundary Conditions

```c
    // BC types for 6 faces (i-, i+, j-, j+, k-, k+)
    PetscInt bctype[6];

    // BC structure
    BcsCtx Bcs;         // Contains BC velocity values
        // Bcs.Ubcs - prescribed velocities at boundaries
```

**BC Type Values:**
- 0: Dirichlet
- 1: Neumann
- 2: Periodic
- 3: Wall (no-slip)
- 4: Inlet with specified profile
- 5: Outlet (zero gradient)

### Linear/Nonlinear Solvers

```c
    // PETSc solvers
    KSP ksp;            // Linear solver (CG, GMRES)
    SNES snes;          // Nonlinear solver

    // Matrices
    Mat A;              // Coefficient matrix
    Mat C, MR, MP;      // Preconditioner matrices

    // Null space (for singular Poisson)
    MatNullSpace nullsp;
```

### Time Integration Parameters

```c
    PetscReal dt;       // Time step size
    PetscReal ren;      // Reynolds number
    PetscReal cfl;      // CFL number (computed)
    PetscReal vnn;      // Viscous number
```

### Statistics (Time Averaging)

```c
    Vec Ucat_sum;           // Σ u over time
    Vec Ucat_cross_sum;     // Σ u'v', v'w', w'u' (Reynolds stress)
    Vec Ucat_square_sum;    // Σ u'u', v'v', w'w'
    PetscInt averaging;     // Number of samples
```

### Multigrid Hierarchy

```c
    // Grid level info
    PetscInt thislevel;     // Current level (0 = finest)
    PetscInt mglevels;      // Total number of levels

    // Linked contexts
    struct UserCtx *user_f; // Finer grid
    struct UserCtx *user_c; // Coarser grid

    // Coarse/fine DM arrays
    DM *da_f, *da_c;
```

---

## 2. Immersed Boundary Nodes: `IBMNodes`

**Defined in:** `Source/variables.h` (lines 146-222)

Stores the surface mesh of immersed bodies.

```c
typedef struct {
    // Mesh connectivity
    PetscInt n_elmt;        // Number of triangular elements
    PetscInt n_v;           // Number of vertices
    PetscInt *nv1, *nv2, *nv3;  // Triangle vertex indices

    // Vertex coordinates
    PetscReal *x_bp, *y_bp, *z_bp;

    // Element properties
    PetscReal *dA;          // Element area
    PetscReal *nf_x, *nf_y, *nf_z;  // Face normals

    // Element centers
    PetscReal *cent_x, *cent_y, *cent_z;

    // Forces on surface
    PetscReal *F_lagr_x, *F_lagr_y, *F_lagr_z;

    // For rotor/blade models
    PetscReal *angle_attack;    // Local angle of attack
    PetscReal *chord_blade;     // Chord length
    PetscReal *CD, *CL;         // Drag/lift coefficients
} IBMNodes;
```

**Data Flow:**
```
surface.ucd  →  Read_ucd()  →  IBMNodes  →  ibm_search_advanced()  →  IBMInfo
(UCD file)       (ibm_io.c)                    (ibm.c)                  │
                                                                        ▼
                                                              ibm_interpolation()
                                                                        │
                                                                        ▼
                                                              Force redistribution
```

---

## 3. IBM Interpolation Info: `IBMInfo`

**Defined in:** `Source/variables.h` (lines 101-144)

Stores interpolation data for transferring values between grid and surface.

```c
typedef struct {
    // Interpolation coefficients
    PetscReal cr[6];        // Trilinear interpolation weights

    // Grid indices for interpolation stencil
    PetscInt i1, j1, k1;    // Lower corner
    PetscInt i2, j2, k2;    // Upper corner (i2 = i1+1, etc.)
    PetscInt ni;            // Local IBM node index

    // Interception point on grid
    PetscReal d_s;          // Distance to surface
    PetscReal x_interp, y_interp, z_interp;  // Interception coordinates
} IBMInfo;
```

**Data Flow:**
```
Grid point  →  Find closest triangle  →  Compute distance  →  Store in IBMInfo
                    │
                    ▼
              Compute interpolation  →  Store coefficients in cr[6]
                    │
                    ▼
              During solve: interpolate velocity to surface
              During solve: spread forces back to grid
```

---

## 4. Fluid-Structure Interaction: `FSInfo`

**Defined in:** `Source/variables.h` (lines 560-601)

Stores structural state and forces for rigid body dynamics.

```c
typedef struct {
    // Position and motion
    Cmpnts S_new, S_old;    // Current and previous position
    Cmpnts S_ang_n, S_ang_o; // Angular position
    Cmpnts U_new, U_old;    // Velocity
    Cmpnts A_new, A_old;    // Acceleration

    // Forces and moments
    PetscReal F_x, F_y, F_z;    // Net force
    PetscReal M_x, M_y, M_z;    // Net moment

    // Structural properties
    PetscReal Mass;             // Total mass
    PetscReal I_x, I_y, I_z;    // Moments of inertia
    PetscReal x_c, y_c, z_c;    // Center of mass

    // Rotor parameters (for wind turbines)
    PetscReal angvel_rotor;     // Angular velocity
    PetscReal Torque_generator; // Generator torque
    PetscReal Power;            // Mechanical power

    // Aitkin acceleration for strong coupling
    PetscReal atk, atk_o;       // Acceleration factors
    Cmpnts dS, dS_o;            // Position corrections
} FSInfo;
```

**Data Flow:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Strong Coupling Loop                      │
│                                                              │
│   ┌─────────────┐                      ┌─────────────┐      │
│   │ Flow Solver │  ──────────────────► │  FSInfo     │      │
│   │             │   F_x, F_y, F_z      │  forces     │      │
│   │  Ucat, P    │   M_x, M_y, M_z      │             │      │
│   └─────────────┘                      └──────┬──────┘      │
│         ▲                                     │              │
│         │                                     ▼              │
│         │                              ┌─────────────┐      │
│         │                              │ Rigid Body  │      │
│         │                              │  Dynamics   │      │
│         │                              │ (6-DOF EOM) │      │
│         │                              └──────┬──────┘      │
│         │                                     │              │
│         │                                     ▼              │
│   ┌─────────────┐                      ┌─────────────┐      │
│   │ IBM Surface │  ◄────────────────── │  FSInfo     │      │
│   │   Update    │   S_new, U_new       │  motion     │      │
│   │             │   (new position)     │             │      │
│   └─────────────┘                      └─────────────┘      │
│                                                              │
│         Repeat until convergence (Aitkin)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Wave Information: `WAVEInfo`

**Defined in:** `Source/variables.h`

Stores wave parameters for two-phase/wave simulations.

```c
typedef struct {
    PetscReal amplitude;    // Wave amplitude
    PetscReal wavenumber;   // Wave number k
    PetscReal frequency;    // Angular frequency omega
    PetscReal phase;        // Phase offset

    // Wave velocity field at inflow
    PetscReal *u_wave, *v_wave, *w_wave;
} WAVEInfo;
```

---

## 6. Helper Structures

### `Cmpnts` - 3-Component Vector

```c
typedef struct {
    PetscReal x, y, z;
} Cmpnts;
```
Used for positions, velocities, forces, and any 3D vector quantity.

### `Cmpnts2` - 2-Component Vector

```c
typedef struct {
    PetscReal x, y;
} Cmpnts2;
```
Used for k-omega values and 2D quantities.

### `Node` - Linked List Node

```c
// In list.h
typedef struct Node {
    PetscInt index;
    struct Node *next;
} Node;
```
Used for dynamic lists during IBM search.

---

## 7. Global vs Local Vectors

### Naming Convention

| Prefix | Meaning | Example |
|--------|---------|---------|
| (none) | Global, distributed | `Ucat`, `P` |
| `l` | Local with ghost points | `lUcat`, `lP` |
| `_o` | Previous timestep | `Ucat_o`, `P_o` |
| `_sum` | Time-averaged | `Ucat_sum` |

### Data Access Pattern

```c
// 1. Scatter global to local (get ghost points)
DMGlobalToLocalBegin(da, Ucat, INSERT_VALUES, lUcat);
DMGlobalToLocalEnd(da, Ucat, INSERT_VALUES, lUcat);

// 2. Get array access
DMDAVecGetArray(fda, lUcat, &ucat);

// 3. Access with stencil (ghosts available)
for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
        for (i=lxs; i<lxe; i++) {
            // Can access ucat[k][j][i-1], ucat[k][j][i+1], etc.
            dudx = (ucat[k][j][i+1].x - ucat[k][j][i-1].x) / (2*dx);
        }
    }
}

// 4. Restore array
DMDAVecRestoreArray(fda, lUcat, &ucat);
```

---

## 8. Data Flow Summary

```
┌───────────────────────────────────────────────────────────────────────┐
│                        INPUT FILES                                     │
│                                                                        │
│   control.xml/dat  →  Simulation parameters (Re, dt, flags)          │
│   grid.dat/xyz.dat →  Grid coordinates → Cent, metrics               │
│   bcs.dat          →  Boundary types → bctype[6]                     │
│   surface.ucd      →  IBM mesh → IBMNodes                            │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     DATA STRUCTURES (UserCtx)                          │
│                                                                        │
│   Grid:     da, fda, Cent, Csi, Eta, Zet, Aj                         │
│   Solution: Ucont, Ucat, P, K_Omega, Levelset                        │
│   Turb:     Cs, Nu_t, F1                                             │
│   BC:       bctype, Bcs                                              │
│   IBM:      ibm (IBMNodes), ibm_intp (IBMInfo)                       │
│   FSI:      fsi (FSInfo)                                             │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        OUTPUT FILES                                    │
│                                                                        │
│   ufield*.dat      →  Cartesian velocity (binary)                    │
│   pfield*.dat      →  Pressure (binary)                              │
│   *.vts            →  VTK structured grid (if enabled)               │
│   statistics.dat   →  Time-averaged fields                           │
│   forces.dat       →  Force history on bodies                        │
└───────────────────────────────────────────────────────────────────────┘
```
