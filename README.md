# Modelisation_Nuage
This repository contains a modelisation of the building "Le nuage de montpellier" in python using Bezier's surface


# Le Nuage - 3D Modeling Project

A mathematical 3D modeling of "Le Nuage" (The Cloud), an iconic building located in Montpellier, France, using Bézier curves and surfaces.

## About

This project recreates the distinctive architecture of Le Nuage building through parametric modeling techniques. The implementation uses Bézier curves and surfaces to generate smooth, organic shapes that characterize the building's unique design.

## Features

- **Bézier Surface Generation**: Creates smooth surfaces using control points and Bézier polynomial interpolation
- **3D Visualization**: Interactive 3D rendering with matplotlib
- **Parametric Modeling**: Mathematical representation of the building's geometry
- **Symmetry Operations**: Central symmetry transformations for mirrored elements
- **Multi-face Rendering**: Separate modeling of different building facades

## Requirements


numpy
matplotlib


Install dependencies:
bash
pip install numpy matplotlib


## Usage

Run the main script:

python Le_Nuage.py


### View Options

Modify the view angle by changing the `ax.view_init()` parameters at the end of the script:

```python
ax.view_init(elev=0, azim=0)   # Front view (small face)
ax.view_init(elev=90, azim=0)  # Top view
ax.view_init(elev=0, azim=90)  # Side view (large face)
```


### Surface Rendering

The building is decomposed into multiple Bézier surface patches (f1, f2, ..., f14 for the main facade, fp1, fp2, ..., fp7 for the side facade), each defined by control points that shape the organic curves of the architecture.

## Mathematical Background

The project implements Bézier surfaces using the formula:

**S(u,v) = Σᵢ Σⱼ Bᵢ,ₙ(u) · Bⱼ,ₘ(v) · Pᵢⱼ**

Where:
- Bᵢ,ₙ(u) are Bernstein basis polynomials
- Pᵢⱼ are control points
- u, v ∈ [0,1] are surface parameters

## Project Structure

- Control point definitions for building geometry
- Bézier curve and surface calculation functions
- Rendering and visualization functions
- Transformation operations (translation, rotation, symmetry)
- Base platform modeling

## Author

Florian Kerbrat

## License

This project is open source and available for educational purposes.

## Acknowledgments

Mathematical modeling inspired by the architectural design of Le Nuage building in Montpellier, France.





