# VFS-Wind Module Interactions and Information Flow

This document describes how different modules in VFS-Wind interact and how
information flows between them during a simulation.

---

## 1. Module Interaction Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN DRIVER (main.c)                            │
│                                                                              │
│  Orchestrates the simulation: initialization → time loop → finalization    │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          │ Calls
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SOLVER MODULES                                     │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  MOMENTUM   │    │  PRESSURE   │    │ TURBULENCE  │    │    FSI      │ │
│  │             │    │             │    │             │    │             │ │
│  │ momentum.c  │◄──►│  poisson.c  │◄──►│   les.c     │◄──►│   fsi.c     │ │
│  │   rhs.c     │    │poisson_hypre│    │ k-omega.c   │    │ fsi_move.c  │ │
│  │ timeadvanc. │    │             │    │wallfunction │    │             │ │
│  │implicitsolver│    │             │    │             │    │             │ │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘ │
│         │                  │                  │                  │         │
│         └──────────────────┼──────────────────┼──────────────────┘         │
│                            │                  │                            │
└────────────────────────────┼──────────────────┼────────────────────────────┘
                             │                  │
          ┌──────────────────┼──────────────────┼──────────────────┐
          │                  │                  │                  │
          ▼                  ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHYSICS MODULES                                     │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │     IBM     │    │  TWO-PHASE  │    │    WAVE     │    │   ROTOR     │ │
│  │             │    │             │    │             │    │             │ │
│  │   ibm.c     │    │   level.c   │    │   wave.c    │    │rotor_model.c│ │
│  │  ibm_io.c   │    │             │    │             │    │             │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
          │                  │                  │                  │
          └──────────────────┼──────────────────┼──────────────────┘
                             │                  │
                             ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE                                       │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   METRICS   │    │    BCS      │    │    I/O      │    │  UTILITIES  │ │
│  │             │    │             │    │             │    │             │ │
│  │  metrics.c  │    │   bcs.c     │    │   data.c    │    │ compgeom.c  │ │
│  │             │    │             │    │ vtk_output  │    │  solvers.c  │ │
│  │             │    │             │    │ xml_input   │    │   init.c    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Information Flow: Single Time Step

### 2.1 Overview of One Timestep

```
ti = current timestep

    ╔════════════════════════════════════════════════════════════════╗
    ║                      START OF TIMESTEP                          ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                                                                 ║
    ║   INPUT STATE:                                                  ║
    ║   • Ucat_o, Ucont_o (velocity at t^n)                          ║
    ║   • P_o (pressure at t^n)                                       ║
    ║   • K_Omega (turbulence at t^n, if RANS)                       ║
    ║   • Levelset (interface at t^n, if two-phase)                  ║
    ║   • FSInfo.S_old (body position at t^n, if FSI)                ║
    ║                                                                 ║
    ╚════════════════════════════════════════════════════════════════╝
                              │
                              ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ STEP 1: Structure Solver (if FSI)                  [fsi.c]     │
    │                                                                 │
    │   Calc_forces_SI() ─────► F_x, F_y, F_z (forces on body)       │
    │                                                                 │
    │   Struc_Solver() ───────► Solve rigid body equations:          │
    │                           m·a = F                               │
    │                           I·α = M                               │
    │                                                                 │
    │   OUTPUT: FSInfo.S_new, FSInfo.U_new (new body state)          │
    │                                                                 │
    │   fsi_move() ───────────► Update IBMNodes coordinates          │
    │   ibm_search() ─────────► Recompute interpolation info         │
    └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ STEP 2: Compute RHS                                [rhs.c]     │
    │                                                                 │
    │   Convection() ─────────► -∇·(u⊗u) (advection term)            │
    │                                                                 │
    │   Viscous() ────────────► ν∇²u (viscous term)                  │
    │                                                                 │
    │   SourceTerms() ────────► IBM forces, buoyancy, wave, rotor    │
    │                                                                 │
    │   OUTPUT: RHS vector (stored for time integration)             │
    └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ STEP 3: Time Advance Momentum           [momentum.c]           │
    │                                         [timeadvancing.c]      │
    │                                         [implicitsolver.c]     │
    │                                                                 │
    │   METHOD SELECTION:                                            │
    │   ├─ Explicit RK:     RungeKutta()                             │
    │   ├─ Implicit RK:     ImpRK() (ESDIRK schemes)                 │
    │   ├─ Adams-Moulton:   AM2 integration                          │
    │   └─ Implicit:        ImplicitMomentumSolver()                 │
    │                                                                 │
    │   OUTPUT: Ucont_star (intermediate velocity, non-divergence-free)
    └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ STEP 4: Pressure Poisson Solve              [poisson.c]        │
    │                                             [poisson_hypre.c]  │
    │                                                                 │
    │   Build RHS: (ρ/Δt) ∇·u*                                       │
    │                                                                 │
    │   Solve: ∇²P = (ρ/Δt) ∇·u*                                     │
    │                                                                 │
    │   SOLVER OPTIONS:                                              │
    │   ├─ PETSc KSP (CG/GMRES)                                      │
    │   ├─ Geometric Multigrid (PoissonSolver_MG)                    │
    │   └─ HYPRE (BoomerAMG preconditioner)                          │
    │                                                                 │
    │   OUTPUT: P (pressure field)                                   │
    └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ STEP 5: Velocity Projection                    [momentum.c]    │
    │                                                                 │
    │   Projection() ─────────► u = u* - (Δt/ρ) ∇P                   │
    │                                                                 │
    │   Contra2Cart() ────────► Convert Ucont → Ucat                 │
    │                                                                 │
    │   OUTPUT: Ucont (divergence-free), Ucat (Cartesian)           │
    └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ STEP 6: Update Turbulence Model                                │
    │                                              [les.c]           │
    │   IF LES:                                                       │
    │   Compute_Smagorinsky_Constant_1() ──► Cs (dynamic constant)   │
    │   Compute_eddy_viscosity_LES() ──────► Nu_t                    │
    │                                                                 │
    │                                              [k-omega.c]       │
    │   IF RANS:                                                      │
    │   Solve_K_Omega() ───────────────────► K_Omega (k, ω)         │
    │   Compute Nu_t from k, ω ────────────► Nu_t                   │
    │                                                                 │
    │   OUTPUT: Nu_t (eddy viscosity for next RHS)                   │
    └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ STEP 7: Two-Phase Interface (if level-set)     [level.c]      │
    │                                                                 │
    │   Advect_Levelset() ────► Move interface with velocity         │
    │                                                                 │
    │   Reinit_Levelset() ────► Maintain signed distance property    │
    │                                                                 │
    │   OUTPUT: Levelset (interface position)                        │
    └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ STEP 8: Output (every tiout steps)                             │
    │                                              [data.c]          │
    │   Binary output: ufield, pfield, nvfield     [vtk_output.cpp]  │
    │   VTK output: *.vts files                                      │
    │   Statistics: time-averaged quantities                         │
    │                                                                 │
    │   Diagnostics:                                                 │
    │   • CFL number                                                 │
    │   • Kinetic energy                                             │
    │   • Convergence residuals                                      │
    └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ╔════════════════════════════════════════════════════════════════╗
    ║                       END OF TIMESTEP                           ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                                                                 ║
    ║   STORE FOR NEXT STEP:                                         ║
    ║   • Ucat_o ← Ucat                                              ║
    ║   • Ucont_o ← Ucont                                            ║
    ║   • P_o ← P                                                    ║
    ║   • FSInfo.S_old ← FSInfo.S_new                                ║
    ║                                                                 ║
    ║   ti++                                                         ║
    ║                                                                 ║
    ╚════════════════════════════════════════════════════════════════╝
```

---

## 3. Module-to-Module Data Flow

### 3.1 Momentum Solver ↔ Pressure Solver

```
                    momentum.c                         poisson.c
               ┌──────────────────┐              ┌──────────────────┐
               │                  │              │                  │
     Ucat_o ──►│  Time advance    │──Ucont*──────│  Poisson solve   │
               │                  │              │                  │
      RHS  ──►│  (intermediate)  │              │  ∇²P = div(u*)   │
               │                  │              │                  │
               └────────┬─────────┘              └────────┬─────────┘
                        │                                 │
                        │                                 P
                        │                                 │
                        ▼                                 ▼
               ┌──────────────────────────────────────────────────────┐
               │                    Projection                         │
               │                                                       │
               │            Ucont = Ucont* - Δt·∇P                    │
               │                                                       │
               └──────────────────────────────────────────────────────┘
                                      │
                                      ▼
                                    Ucat (via Contra2Cart)
```

### 3.2 Flow Solver ↔ IBM Module

```
        Flow Solver                          IBM Module
    ┌─────────────────┐                 ┌─────────────────┐
    │                 │                 │                 │
    │  Ucat, P       ─┼─────────────────┼►  Interpolate   │
    │  (flow field)   │                 │  to surface     │
    │                 │                 │                 │
    │                 │                 │  u_surf = Σ u_i·w_i
    │                 │                 │                 │
    │                 │   F_ibm         │  Compute force  │
    │  Apply force   ◄┼─────────────────┼─                │
    │  to grid        │                 │  (no-slip BC)   │
    │                 │                 │                 │
    └─────────────────┘                 └─────────────────┘

    Key functions:
    ibm.c:
      ibm_interpolation_advanced()  →  Interpolate grid values to surface
      Calc_F_lagr()                 →  Compute Lagrangian forces
      Spread_force()                →  Distribute force to Eulerian grid
```

### 3.3 Flow Solver ↔ FSI Module

```
        Flow Solver                          FSI Module
    ┌─────────────────┐                 ┌─────────────────┐
    │                 │                 │                 │
    │                 │  F_x, F_y, F_z  │                 │
    │  Compute forces─┼────────────────►│  Rigid body     │
    │  (pressure +    │  M_x, M_y, M_z  │  dynamics       │
    │   shear stress) │                 │                 │
    │                 │                 │  m·a = F        │
    │                 │                 │  I·α = M        │
    │                 │                 │                 │
    │                 │  S_new, U_new   │                 │
    │  Update IBM    ◄┼─────────────────┼─ New position   │
    │  mesh position  │                 │  & velocity     │
    │                 │                 │                 │
    └─────────────────┘                 └─────────────────┘

    STRONG COUPLING (iterative):
    ┌───────────────────────────────────────────────────────────────┐
    │ for (itr = 1; itr <= SC_MAX; itr++) {                        │
    │     1. Solve flow with current body position                 │
    │     2. Compute forces on body                                │
    │     3. Solve structure with new forces                       │
    │     4. Update body position                                  │
    │     5. Check convergence (Aitkin acceleration)               │
    │ }                                                            │
    └───────────────────────────────────────────────────────────────┘
```

### 3.4 Turbulence ↔ Momentum

```
     Momentum Solver                      Turbulence Module
    ┌─────────────────┐                 ┌─────────────────┐
    │                 │                 │                 │
    │                 │     Ucat        │                 │
    │  Velocity field─┼────────────────►│  Compute        │
    │                 │                 │  strain rate S  │
    │                 │                 │                 │
    │                 │    Nu_t         │  LES: ν_t = (CsΔ)²|S|
    │  Viscous term  ◄┼─────────────────┼─                │
    │  uses ν_eff     │                 │  RANS: ν_t = k/ω│
    │  = ν + ν_t      │                 │                 │
    │                 │                 │                 │
    └─────────────────┘                 └─────────────────┘
```

### 3.5 Level-Set ↔ Flow/Properties

```
     Level-Set Module                     Flow Properties
    ┌─────────────────┐                 ┌─────────────────┐
    │                 │                 │                 │
    │  φ (level-set)  ─┼────────────────►│  Density:       │
    │                 │                 │  ρ = ρ₁H(φ) +   │
    │  Interface at   │                 │      ρ₂(1-H(φ)) │
    │  φ = 0          │                 │                 │
    │                 │                 │  Viscosity:     │
    │                 │                 │  μ = μ₁H(φ) +   │
    │                 │                 │      μ₂(1-H(φ)) │
    │                 │                 │                 │
    │  Advection:     │     Ucat        │  Surface tension│
    │  ∂φ/∂t + u·∇φ=0◄┼─────────────────┼  σκn at φ=0    │
    │                 │                 │                 │
    └─────────────────┘                 └─────────────────┘
```

---

## 4. Function Call Chains

### 4.1 Main Time Loop (main.c)

```c
main() {
    PetscInitialize();

    // Parse configuration
    if (xml_input) ParseXML();
    else PetscOptionsInsertFile("control.dat");

    // Initialize
    MG_Initial(&usermg);           // Multigrid setup
    FormMetrics(&usermg);          // Grid metrics
    FormInitialize(&usermg);       // Initial conditions

    // Time loop
    for (ti = tistart; ti < totalsteps; ti++) {

        if (fsi) {
            Struc_Solver(&usermg);     // FSI: structure solver
        }

        // Strong coupling iterations
        for (itr = 1; itr <= SC_MAX; itr++) {

            Flow_Solver(&usermg);       // Momentum + pressure

            if (fsi) {
                Check_FSI_Convergence();
            }
        }

        // Post-processing
        if (ti % tiout == 0) {
            Output_Fields(&usermg);
        }
    }

    PetscFinalize();
}
```

### 4.2 Flow Solver Chain

```c
Flow_Solver(user) {

    // Step 1: Compute right-hand side
    ComputeRHS(user);                     // rhs.c
        ├── Convection(user);             // Advection terms
        ├── Viscous(user);                // Viscous diffusion
        └── SourceTerms(user);            // IBM, buoyancy, etc.

    // Step 2: Time advance
    if (explicit_rk) {
        RungeKutta(user);                 // timeadvancing.c
    } else if (implicit_rk) {
        ImpRK(user);                      // implicitsolver.c
    } else if (am2) {
        AM2_Advance(user);                // momentum.c
    } else {
        ImplicitMomentumSolver(user);     // implicitsolver.c
    }

    // Step 3: Pressure correction
    PoissonSolver_MG(user);               // poisson.c
        ├── PoissonRHS2(user);            // Build RHS: div(u*)
        └── KSPSolve(ksp, ...);           // Solve ∇²P = RHS

    // Step 4: Projection
    Projection(user);                     // momentum.c
        └── Ucont = Ucont* - dt * grad(P)

    // Step 5: Transform coordinates
    Contra2Cart(user);                    // Convert Ucont → Ucat

    // Step 6: Update turbulence
    if (les) {
        Compute_Smagorinsky_Constant_1(user);  // les.c
        Compute_eddy_viscosity_LES(user);
    }
    if (rans) {
        Solve_K_Omega(user);              // k-omega.c
    }

    // Step 7: Apply BCs
    FormBCS(user);                        // bcs.c
}
```

### 4.3 IBM Processing Chain

```c
// During initialization (main.c)
Read_ucd(ibm, "surface.ucd");             // ibm_io.c: read mesh
ibm_search_advanced(user, ibm);           // ibm.c: find neighbors
ibm_interpolation_setup(user, ibm);       // Setup interpolation

// During each timestep
ibm_interpolation(user, ibm);             // Interpolate u to surface
    │
    └── for each IBM point:
            u_surf[i] = Σ cr[j] * u_grid[stencil[j]]

Calc_F_lagr(user, ibm);                   // Compute forces
    │
    └── F_lagr = (u_desired - u_surf) / dt   (direct forcing)

Spread_IBM_Force(user, ibm);              // Apply to grid
    │
    └── for each IBM point:
            f_grid[stencil[j]] += cr[j] * F_lagr[i]
```

---

## 5. Key Function Reference

| Module | Function | Purpose |
|--------|----------|---------|
| **main.c** | `main()` | Entry point, time loop |
| | `MG_Initial()` | Initialize multigrid levels |
| **momentum.c** | `Projection()` | Velocity correction with pressure |
| | `Contra2Cart()` | Transform contravariant → Cartesian |
| **rhs.c** | `ComputeRHS()` | Build momentum equation RHS |
| | `Convection()` | Advection term |
| | `Viscous()` | Viscous diffusion term |
| **poisson.c** | `PoissonSolver_MG()` | Multigrid pressure solver |
| | `PoissonRHS2()` | Build Poisson RHS (divergence) |
| **timeadvancing.c** | `RungeKutta()` | Explicit RK time advance |
| **implicitsolver.c** | `ImplicitMomentumSolver()` | Implicit time advance |
| | `ImpRK()` | ESDIRK schemes |
| **les.c** | `Compute_Smagorinsky_Constant_1()` | Dynamic Cs |
| | `Compute_eddy_viscosity_LES()` | LES eddy viscosity |
| **k-omega.c** | `Solve_K_Omega()` | RANS k-ω SST solver |
| | `K_Omega_IC()` | Initial conditions |
| **ibm.c** | `ibm_search_advanced()` | Find IBM neighbors |
| | `ibm_interpolation_advanced()` | Grid-to-surface transfer |
| **ibm_io.c** | `Read_ucd()` | Read IBM surface mesh |
| **fsi.c** | `Calc_forces_SI()` | Compute forces on body |
| | `Struc_Solver()` | Rigid body dynamics |
| **fsi_move.c** | `fsi_move()` | Update body position |
| **level.c** | `Advect_Levelset()` | Advect interface |
| | `Reinit_Levelset()` | Redistance level-set |
| **bcs.c** | `FormBCS()` | Apply boundary conditions |
| **metrics.c** | `FormMetrics()` | Compute grid metrics |
| **wallfunction.c** | `wall_function()` | Wall model |
| **rotor_model.c** | `Compute_Rotor_Source()` | Actuator disk/line |
