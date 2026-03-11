# VFS-Wind Mesh Generator

## Overview

The VFS-Wind Mesh Generator creates structured curvilinear meshes from XML specifications. It eliminates the need for external mesh generation tools for common CFD geometries while supporting complex body-fitted and terrain-following meshes.

## Features

- **Cartesian meshes** with optional stretching
- **Channel flows** with wall clustering (tanh, geometric, hyperbolic)
- **Body-fitted meshes** for wavy/bumpy walls
- **Terrain-following meshes** with multiple Gaussian hills
- **Cylindrical/pipe** coordinates
- **Expression-based surfaces** using mathematical functions
- **Automatic quality metrics** computation

## XML Configuration

### Basic Structure

```xml
<vfswind version="1.0">
  <grid type="channel">
    <dimensions ni="256" nj="65" nk="128"/>
    <domain lx="10.0" ly="2.0" lz="3.14"/>
    <stretching direction="j" type="tanh" factor="2.5"/>
    <bottom_wall type="wavy">
      <amplitude>0.05</amplitude>
      <wavelength_x>1.0</wavelength_x>
    </bottom_wall>
    <top_wall y="2.0"/>
  </grid>
</vfswind>
```

### Mesh Types

| Type | Description | Use Case |
|------|-------------|----------|
| `cartesian` | Simple Cartesian grid | Box domains |
| `channel` | Channel with body-fitted walls | Turbulent channels, bumpy walls |
| `terrain` | Terrain-following coordinates | Atmospheric flows, hills |
| `cylindrical` | Cylindrical/pipe coordinates | Pipe flows, annular ducts |
| `o-grid` | O-grid around body | External aerodynamics |
| `curvilinear` | General mapping | Complex geometries |

---

## Stretching Functions

### Available Types

#### Uniform (`uniform`)
No stretching - linear distribution.

#### Hyperbolic Tangent (`tanh`)
Symmetric clustering at both ends. Best for channel flows.

```xml
<stretching direction="j" type="tanh" factor="2.5"/>
```

**Parameters:**
- `factor`: Controls clustering intensity (1.0 = mild, 3.0 = strong)

**Formula:**
```
s = 0.5 * (1 + tanh(δ(2η - 1)) / tanh(δ))
where δ = factor × atanh(0.99)
```

#### Geometric (`geometric`)
Constant cell-to-cell growth ratio. Good for boundary layers.

```xml
<stretching direction="j" type="geometric" ratio="1.08" near="min"/>
```

**Parameters:**
- `ratio`: Growth ratio between cells (1.05-1.15 typical)
- `near`: Cluster near "min" or "max" boundary

#### Hyperbolic / Vinokur (`hyperbolic`)
Smooth clustering with specified first cell size.

```xml
<stretching direction="j" type="hyperbolic" first_cell="1e-5"/>
```

**Parameters:**
- `first_cell`: Size of first cell at wall (in physical units)

#### Two-Sided (`two-sided`)
Clusters at both ends independently.

```xml
<stretching direction="j" type="two-sided" factor="2.0"/>
```

---

## Surface Definitions

### Flat Surface

```xml
<bottom_wall type="flat" y="0.0"/>
<top_wall type="flat" y="2.0"/>
```

### Wavy/Sinusoidal Wall

```xml
<bottom_wall type="wavy">
  <amplitude>0.05</amplitude>
  <wavelength_x>1.0</wavelength_x>
  <wavelength_z>0.0</wavelength_z>  <!-- 0 = 2D wave -->
  <phase>0.0</phase>
</bottom_wall>
```

**Surface equation:** `h(x,z) = A × sin(2πx/λx + φ) × sin(2πz/λz)`

### Gaussian Hills

```xml
<surface type="gaussian">
  <hill x="5.0" z="5.0" height="0.5" sigma_x="1.0" sigma_z="1.0"/>
  <hill x="12.0" z="5.0" height="0.8" sigma_x="1.5" sigma_z="1.0"/>
</surface>
```

**Surface equation:** `h(x,z) = Σ Hi × exp(-(x-xi)²/2σx² - (z-zi)²/2σz²)`

### Mathematical Expression

```xml
<bottom_wall type="function">
  <expression>0.1 * sin(2*pi*x/2.0) * exp(-0.1*x)</expression>
</bottom_wall>
```

**Supported functions:**
- Trigonometric: `sin`, `cos`, `tanh`
- Exponential: `exp`, `sqrt`
- Operators: `+`, `-`, `*`, `/`, `^` (power)
- Constants: `pi`
- Variables: `x`, `z`

### File-Based Surface

```xml
<surface type="file" file="terrain_elevation.dat"/>
```

---

## Example Configurations

### 1. Standard Channel Flow

```xml
<grid type="channel">
  <dimensions ni="128" nj="65" nk="128"/>
  <domain lx="6.28" ly="2.0" lz="3.14"/>
  <stretching direction="j" type="tanh" factor="2.5"/>
  <bottom_wall y="0.0"/>
  <top_wall y="2.0"/>
</grid>
```

### 2. Channel with Wavy Bottom Wall

```xml
<grid type="channel">
  <dimensions ni="256" nj="65" nk="128"/>
  <domain lx="10.0" ly="2.0" lz="3.14"/>
  <stretching direction="j" type="tanh" factor="2.0"/>
  <bottom_wall type="wavy">
    <amplitude>0.05</amplitude>
    <wavelength_x>1.0</wavelength_x>
  </bottom_wall>
  <top_wall y="2.0"/>
</grid>
```

### 3. NASA 2D Bump Case

```xml
<grid type="channel">
  <dimensions ni="512" nj="129" nk="1"/>
  <domain lx="3.0" ly="0.5"/>
  <stretching direction="j" type="hyperbolic" first_cell="1e-5"/>
  <bottom_wall type="function">
    <expression>0.05 * exp(-25*(x-1.5)^2)</expression>
  </bottom_wall>
  <top_wall y="0.5"/>
</grid>
```

### 4. Terrain with Multiple Hills

```xml
<grid type="terrain">
  <dimensions ni="256" nj="97" nk="256"/>
  <domain lx="20.0" ly="10.0" lz="20.0"/>
  <surface type="gaussian">
    <hill x="5.0" z="10.0" height="0.5" sigma_x="1.0" sigma_z="1.0"/>
    <hill x="15.0" z="10.0" height="1.0" sigma_x="2.0" sigma_z="1.5"/>
  </surface>
  <top y="10.0"/>
  <blending height="5.0"/>
  <stretching direction="j" type="tanh" factor="1.5"/>
</grid>
```

### 5. Pipe Flow

```xml
<grid type="cylindrical">
  <dimensions ni="64" nj="65" nk="256"/>  <!-- theta, r, z -->
  <geometry r_inner="0.0" r_outer="1.0" length="10.0"/>
  <stretching direction="j" type="geometric" ratio="1.05" near="outer"/>
</grid>
```

---

## Mesh Quality Metrics

The generator automatically computes and reports mesh quality:

```
=== Mesh Quality Report ===
  Min Jacobian:       2.34e-04
  Max Jacobian:       1.52e-02
  Max Aspect Ratio:   125.30
  Min Orthogonality:  72.30 degrees
  Negative Cells:     0
===========================
```

**Quality Guidelines:**

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Min Jacobian | > 0 | > 0 | ≤ 0 (invalid) |
| Aspect Ratio | < 100 | < 500 | > 1000 |
| Orthogonality | > 60° | > 45° | < 30° |
| Skewness | < 0.5 | < 0.85 | > 0.95 |

---

## Output Formats

The generator can write meshes in multiple formats:

### Plot3D XYZ (default)
```xml
<output file="grid.xyz" format="xyz"/>
```
Compatible with existing VFS-Wind grid reader.

### VTK Structured Grid
```xml
<output file="mesh.vtk" format="vtk"/>
```
For visualization in ParaView.

---

## Integration with VFS-Wind

### Option 1: Generate mesh on-the-fly

```xml
<vfswind version="1.0">
  <grid type="channel">
    <!-- ... mesh specification ... -->
  </grid>

  <!-- Rest of simulation config -->
  <simulation>
    <timestep dt="0.001" totalsteps="1000"/>
  </simulation>
</vfswind>
```

### Option 2: Generate mesh and save to file

Use the mesh generator to create a grid file, then reference it:

```xml
<grid file="generated_mesh.xyz" format="xyz"/>
```

---

## Algorithm Details

### Transfinite Interpolation (Algebraic)

For body-fitted meshes, we use transfinite interpolation:

```
x(ξ,η,ζ) = (1-η)×x_bottom(ξ,ζ) + η×x_top(ξ,ζ)
```

where:
- `ξ, η, ζ ∈ [0,1]` are computational coordinates
- Stretching is applied to `η` to cluster points near walls

### Terrain-Following (Gal-Chen)

For terrain meshes, we use Gal-Chen coordinates with smooth blending:

```
y(x,η,z) = h(x,z) + η×(H - h(x,z)) × blend(η)
```

where:
- `h(x,z)` is terrain height
- `H` is domain top
- `blend(η)` smoothly transitions from terrain-following to flat

---

## Future Enhancements

- [ ] O-grid generation for cylinders and airfoils
- [ ] Elliptic smoothing for improved orthogonality
- [ ] Multi-block structured grids
- [ ] CGNS file format support
- [ ] Spline-based surface definitions
- [ ] Automatic refinement near surfaces

---

## References

1. Vinokur, M. "On One-Dimensional Stretching Functions for Finite-Difference Calculations", J. Comput. Phys., 1983.
2. Thompson, J.F., Warsi, Z.U.A., Mastin, C.W., "Numerical Grid Generation", North-Holland, 1985.
3. Gal-Chen, T., Somerville, R.C.J., "On the use of a coordinate transformation for the solution of the Navier-Stokes equations", J. Comput. Phys., 1975.
