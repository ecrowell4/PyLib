# PyLib (nlopy)
A module useful for nonlinear optics. 

# Install 
To install PyLib (i.e. nlopy), download this repository. Then navigate to the directory above nlopy and run `python setup.py install`. This should allow you to import nlopy
like any other python module.

# Primary Modules
nlopy is comprised of several modules, each with it's own focus. Here we list each module and briefly describe it's focus.

## quantum_solvers
The quantum_solvers module contains several utilities for solving the Schr&ouml;dinger equation. Primary files are solver_utils.py, solver_1D.py, solver_2D.py, evolver_utils.py,
evolver_1D.py, many_electron_utils.py, evolver_HF, and State.py

## hyperpol
This module provides utilities for computing the polarizability and hyperpolarizabilities of quantum systems. 

### sum_over_states
Various utilities for computing the (hyper)polarizabilities from the sum-over-states (SOS) expressions from perturbation theory.

### finite_fields
Not yet implemented. Will provide utility for using finite fields to compute the (hyper)polarizabilities.

## monte_carlo
This module provides utilities for Monte Carlo sampling of the parameter spaces relevant to the optical polarizability and hyperpolarizabilities.

## plotpy
Primary file is PlotFormat.py, which sets pyplot rcparams for pretty graphs. 
