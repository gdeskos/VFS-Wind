# VFS-Wind Numerical Algorithms

This document describes the numerical algorithms and methods implemented in VFS-Wind.

---

## 1. Governing Equations

VFS-Wind solves the incompressible Navier-Stokes equations:

```
Momentum:    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f

Continuity:  ∇·u = 0
```

Where:
- `u` = velocity vector (u, v, w)
- `p` = pressure
- `ρ` = density
- `ν` = kinematic viscosity
- `f` = body forces (IBM, buoyancy, rotor, etc.)

---

## 2. Spatial Discretization

### 2.1 Curvilinear Coordinates

The code uses a structured, curvilinear grid with body-fitted coordinates:

```
Physical space: (x, y, z)
Computational space: (ξ, η, ζ)

Transformation: x = x(ξ, η, ζ), y = y(ξ, η, ζ), z = z(ξ, η, ζ)
```

**Metrics (computed in `metrics.c`):**

```
┌───────────────────────────────────────────────────────────────┐
│ Jacobian:  J = ∂(x,y,z)/∂(ξ,η,ζ)                             │
│                                                               │
│ Contravariant base vectors:                                   │
│                                                               │
│   ξ̂ = (1/J) × (∂r/∂η × ∂r/∂ζ)   →  stored in Csi           │
│   η̂ = (1/J) × (∂r/∂ζ × ∂r/∂ξ)   →  stored in Eta           │
│   ζ̂ = (1/J) × (∂r/∂ξ × ∂r/∂η)   →  stored in Zet           │
│                                                               │
│ Where r = (x, y, z) is the position vector                   │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 Contravariant vs Cartesian Velocity

**Contravariant velocity (Ucont):**
- Face-normal flux components
- Used for mass conservation
- `U^ξ = u·ξ̂`, `U^η = u·η̂`, `U^ζ = u·ζ̂`

**Cartesian velocity (Ucat):**
- Physical velocity components
- Used for momentum and output
- (u, v, w)

**Conversion (`Contra2Cart` in momentum.c):**
```
u = (U^ξ × ξ̂ + U^η × η̂ + U^ζ × ζ̂)
```

### 2.3 Finite Volume Discretization

```
        j+1  ┌────────────────┐
             │                │
             │    Cell (i,j)  │
             │       ●        │  ● = cell center (scalars: p, T, k, ω)
             │                │
        j    ├────────────────┤
             │                │
             U^η face (i,j)   │  U^η = face-normal flux
             │                │
        j-1  └────────────────┘
            i-1      i       i+1

Fluxes computed at faces, scalars stored at cell centers
```

---

## 3. Time Integration

### 3.1 Projection Method (Fractional Step)

The velocity-pressure coupling uses a projection method:

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Compute intermediate velocity u*                        │
│                                                                  │
│   u* - u^n                                                       │
│   ────────  = -∇p^n + RHS(u^n)                                  │
│      Δt                                                          │
│                                                                  │
│   where RHS = -(u·∇)u + ν∇²u + f                                │
│                                                                  │
│   Note: u* is NOT divergence-free                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Solve pressure Poisson equation                         │
│                                                                  │
│   ∇²φ = (1/Δt) ∇·u*                                             │
│                                                                  │
│   where φ = p^{n+1} - p^n  (pressure correction)                │
│                                                                  │
│   This ensures ∇·u^{n+1} = 0                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Correct velocity (projection)                           │
│                                                                  │
│   u^{n+1} = u* - Δt ∇φ                                          │
│                                                                  │
│   p^{n+1} = p^n + φ                                             │
│                                                                  │
│   Result: u^{n+1} is divergence-free                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Explicit Runge-Kutta (timeadvancing.c)

**3-stage, 2nd-order scheme:**

```
Stage 1:  u^(1) = u^n + Δt · RHS(u^n)

Stage 2:  u^(2) = (3/4)u^n + (1/4)u^(1) + (1/4)Δt · RHS(u^(1))

Stage 3:  u^{n+1} = (1/3)u^n + (2/3)u^(2) + (2/3)Δt · RHS(u^(2))
```

### 3.3 Implicit Runge-Kutta - ESDIRK (implicitsolver.c)

**ESDIRK2 (2nd-order A-stable):**

```
Coefficients:
γ = 1 - √2/2 ≈ 0.2929

Butcher tableau:
    0   │  0        0
    2γ  │  γ        γ
    1   │  1-γ      γ
    ────┼─────────────
        │  1-γ      γ

Each stage requires solving:
  (I - γΔt·L) u^(k) = u^n + Δt · explicit_terms
```

### 3.4 Adams-Moulton 2nd Order (AM2)

```
u^{n+1} = u^n + (Δt/2) [3·RHS^n - RHS^{n-1}]

Requires storing RHS from previous timestep
```

---

## 4. Pressure Poisson Solver

### 4.1 Discrete Poisson Equation (poisson.c)

```
In computational coordinates:

∇·(∇φ) ≈ (1/J) [ ∂/∂ξ(J g^ξξ ∂φ/∂ξ) + ∂/∂η(J g^ηη ∂φ/∂η) + ∂/∂ζ(J g^ζζ ∂φ/∂ζ)
               + cross terms with g^ξη, g^ηζ, g^ξζ ]

where g^ij are metric tensor components
```

### 4.2 Multigrid V-Cycle (PoissonSolver_MG)

```
┌─────────────────────────────────────────────────────────────────┐
│                      V-CYCLE MULTIGRID                           │
│                                                                  │
│  Level 0 (finest)                                               │
│      │                                                          │
│      │ Pre-smooth (ν₁ iterations of smoother)                   │
│      ▼                                                          │
│      Restrict residual to coarser grid                          │
│      │                                                          │
│  Level 1                                                        │
│      │                                                          │
│      │ Pre-smooth                                               │
│      ▼                                                          │
│      Restrict residual                                          │
│      │                                                          │
│  Level 2 (coarsest)                                             │
│      │                                                          │
│      ▼                                                          │
│      Solve exactly (direct solver or many iterations)           │
│      │                                                          │
│      ▼                                                          │
│  Level 1                                                        │
│      │                                                          │
│      │ Prolongate correction from coarser grid                  │
│      │ Post-smooth (ν₂ iterations)                              │
│      ▼                                                          │
│  Level 0                                                        │
│      │                                                          │
│      │ Prolongate correction                                    │
│      │ Post-smooth                                              │
│      ▼                                                          │
│      Solution                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key functions:**
- `Restriction()` in `solvers.c`: Fine → coarse
- `Prolongation()` in `solvers.c`: Coarse → fine
- Smoother: Gauss-Seidel or SOR

### 4.3 HYPRE Preconditioner (poisson_hypre.c)

Uses HYPRE BoomerAMG (algebraic multigrid) as preconditioner for PETSc KSP.

---

## 5. Turbulence Modeling

### 5.1 Large Eddy Simulation (les.c)

**Smagorinsky Model:**

```
Subgrid stress:  τ_ij = -2ν_t S_ij

Eddy viscosity:  ν_t = (C_s Δ)² |S|

where:
  S_ij = (1/2)(∂u_i/∂x_j + ∂u_j/∂x_i)  (strain rate tensor)
  |S| = √(2 S_ij S_ij)                   (strain rate magnitude)
  Δ = (Δx Δy Δz)^{1/3}                   (filter width)
  C_s ≈ 0.1 (static) or computed dynamically
```

**Dynamic Smagorinsky (`Compute_Smagorinsky_Constant_1`):**

```
Uses test filter at scale 2Δ to compute optimal C_s:

C_s² = <L_ij M_ij> / <M_ij M_ij>

where:
  L_ij = û_i û_j - (u_i u_j)^           (resolved stress)
  M_ij = 2Δ²|S̃|S̃_ij - 2(2Δ)²|S̃|S̃_ij   (test-filter terms)
  ^ denotes test filtering
```

### 5.2 RANS k-ω SST (k-omega.c)

**Transport equations:**

```
k-equation:
  ∂k/∂t + u·∇k = P_k - β*ωk + ∇·[(ν + σ_k ν_t)∇k]

ω-equation:
  ∂ω/∂t + u·∇ω = α(P_k/ν_t) - βω² + ∇·[(ν + σ_ω ν_t)∇ω] + CD_kω

Eddy viscosity:
  ν_t = a₁k / max(a₁ω, SF₂)

where S = |S_ij| and F₂ is blending function
```

**SST Blending Functions:**

```
F₁ = tanh(arg₁⁴)
arg₁ = min[max(√k/(β*ωy), 500ν/(y²ω)), 4σ_ω₂k/(CD_kω y²)]

F₂ = tanh(arg₂²)
arg₂ = max(2√k/(β*ωy), 500ν/(y²ω))

Coefficients blend between k-ω (near wall) and k-ε (far field)
```

### 5.3 Wall Functions (wallfunction.c)

**Log-law formulation:**

```
u⁺ = y⁺               for y⁺ < 11.63  (viscous sublayer)
u⁺ = (1/κ)ln(y⁺) + B  for y⁺ > 11.63  (log layer)

where:
  u⁺ = u/u_τ
  y⁺ = y·u_τ/ν
  κ = 0.41 (von Kármán constant)
  B = 5.2
  u_τ = √(τ_w/ρ) (friction velocity)
```

**Cabot wall model (`wall_function` in wallfunction.c):**

Iteratively solves for u_τ matching the first grid point velocity.

---

## 6. Immersed Boundary Method

### 6.1 Direct Forcing IBM (ibm.c)

```
┌─────────────────────────────────────────────────────────────────┐
│                    IBM ALGORITHM                                 │
│                                                                  │
│  1. SEARCH PHASE (preprocessing):                               │
│     - For each Lagrangian point, find enclosing Eulerian cells  │
│     - Store interpolation stencil and coefficients              │
│                                                                  │
│  2. INTERPOLATION PHASE (each timestep):                        │
│     u_lag = Σ w_i · u_eul[i]                                    │
│     (interpolate Eulerian velocity to Lagrangian points)        │
│                                                                  │
│  3. FORCING PHASE:                                              │
│     f_lag = (u_desired - u_lag) / Δt                            │
│     (compute force needed to enforce BC)                        │
│                                                                  │
│  4. SPREADING PHASE:                                            │
│     f_eul[i] += w_i · f_lag                                     │
│     (distribute Lagrangian force to Eulerian grid)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Interpolation/Spreading

**Trilinear interpolation:**

```
u(x,y,z) = Σ_{i,j,k ∈ stencil} c_ijk · u[i,j,k]

where c_ijk are the trilinear weights:
  c_ijk = (1-|ξ-i|)(1-|η-j|)(1-|ζ-k|)
```

**Delta function spreading:**

```
For regularized delta function δ_h:

  f_eul(x) = Σ_lag f_lag · δ_h(x - x_lag) · ΔV_lag

Options (deltafunc parameter):
  - Point collocation
  - 1-point, 3-point, or Gaussian delta
```

---

## 7. Level-Set Method (level.c)

### 7.1 Interface Representation

```
φ(x,t) = signed distance function

  φ > 0  →  Fluid 1 (e.g., water)
  φ = 0  →  Interface
  φ < 0  →  Fluid 2 (e.g., air)

Properties vary across interface:
  ρ(φ) = ρ₁H(φ) + ρ₂(1-H(φ))
  μ(φ) = μ₁H(φ) + μ₂(1-H(φ))

where H(φ) = smoothed Heaviside function
```

### 7.2 Advection

```
∂φ/∂t + u·∇φ = 0

Discretization: WENO5 or 3rd-order upwind
```

### 7.3 Reinitialization

To maintain signed distance property |∇φ| = 1:

```
∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0

Iterate to steady state in pseudo-time τ
```

---

## 8. Fluid-Structure Interaction (fsi.c)

### 8.1 Rigid Body Dynamics

**6-DOF equations of motion:**

```
Translation:  m·a_c = F_fluid + F_external

Rotation:     I·α + ω × (I·ω) = M_fluid + M_external

where:
  m = mass
  a_c = acceleration of center of mass
  I = inertia tensor
  α = angular acceleration
  ω = angular velocity
```

### 8.2 Force Computation (Calc_forces_SI)

```
F_fluid = ∫_surface (p·n + τ·n) dA

where:
  p = pressure
  n = surface normal
  τ = viscous stress tensor
```

### 8.3 Time Integration (Newmark-β)

```
S^{n+1} = S^n + Δt·V^n + (Δt²/2)[(1-2β)A^n + 2β·A^{n+1}]

V^{n+1} = V^n + Δt·[(1-γ)A^n + γ·A^{n+1}]

with β = 1/4, γ = 1/2 (average acceleration method)
```

### 8.4 Strong Coupling with Aitkin Acceleration

```
for iteration k = 1, 2, ... until convergence:

    1. Solve flow with body at position S^(k)
    2. Compute forces F^(k)
    3. Solve structure: A^(k+1) = F^(k)/m
    4. Predict new position: S^(k+1)_pred

    5. Aitkin acceleration:
       ω^(k) = ω^(k-1) · [ΔS^(k-1)·(ΔS^(k-1) - ΔS^(k))] / |ΔS^(k-1) - ΔS^(k)|²
       S^(k+1) = S^(k) + ω^(k) · (S^(k+1)_pred - S^(k))

    6. Check: |S^(k+1) - S^(k)| < tolerance ?
```

---

## 9. Wind Turbine Rotor Models (rotor_model.c)

### 9.1 Actuator Disk Model

```
Uniform thrust force over rotor disk:

  T = (1/2) ρ A_rotor C_T U_∞²

Force per unit volume in disk region:
  f = T / V_disk

where:
  C_T = thrust coefficient (from BEM theory or specified)
  U_∞ = freestream velocity
```

### 9.2 Actuator Line Model

```
Per blade section at radius r:

1. Compute relative velocity:
   U_rel = U_wind - Ω × r

2. Compute angle of attack:
   α = arctan(U_axial / U_tangential) - β
   where β = blade pitch + twist

3. Look up lift/drag coefficients:
   C_L(α), C_D(α) from airfoil tables

4. Compute sectional forces:
   L = (1/2) ρ |U_rel|² c C_L    (lift)
   D = (1/2) ρ |U_rel|² c C_D    (drag)

5. Distribute forces to Eulerian grid using spreading function
```

---

## 10. Boundary Conditions (bcs.c)

### 10.1 BC Types

| Code | Type | Implementation |
|------|------|----------------|
| 0 | Dirichlet | u = u_specified |
| 1 | Neumann | ∂u/∂n = 0 |
| 2 | Periodic | u(x=0) = u(x=L) |
| 3 | Wall | u = 0 (no-slip) |
| 4 | Inlet | u = profile(y,z) |
| 5 | Outlet | ∂u/∂n = 0 (convective) |

### 10.2 Inlet Profiles

```
// Uniform:
u = U_inlet, v = 0, w = 0

// Power law:
u = U_ref * (y/δ)^{1/n}

// Log law:
u = (u_τ/κ) * ln(y/y_0)

// Recycling (turbulent inflow):
u(inlet) = u(x_recycle) - <u(x_recycle)> + U_inlet
```

---

## 11. Stability Criteria

### 11.1 CFL Condition

```
CFL = max(|u|/Δx + |v|/Δy + |w|/Δz) · Δt < CFL_max

where CFL_max ≈ 1.0 for explicit schemes
```

### 11.2 Viscous Stability

```
VNN = ν · Δt / min(Δx², Δy², Δz²) < VNN_max

where VNN_max ≈ 0.5 for explicit viscous terms
```

---

## 12. Parallel Implementation

### 12.1 Domain Decomposition

```
PETSc DMDA provides automatic decomposition:

┌─────────┬─────────┬─────────┐
│  Proc 0 │  Proc 1 │  Proc 2 │
│         │         │         │
├─────────┼─────────┼─────────┤
│  Proc 3 │  Proc 4 │  Proc 5 │
│         │         │         │
└─────────┴─────────┴─────────┘

Each processor owns a subdomain with ghost points for stencil operations
```

### 12.2 Communication Pattern

```
1. Global to Local scatter:
   DMGlobalToLocalBegin/End()
   - Updates ghost points from neighbors

2. Local computation:
   - Each processor computes on its subdomain
   - Can access ghost points for stencil

3. Local to Global scatter (if needed):
   DMLocalToGlobalBegin/End()
```

### 12.3 Global Reductions

```
// Sum over all processors
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

// Max over all processors
MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
```
