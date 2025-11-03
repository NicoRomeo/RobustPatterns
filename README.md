# RobustPatterns
Lattice, PDE and transfer matrix code to study robust patterning, supporting the paper 'Information bounds the robustness of self-organized systems' by N. Romeo, D.G. Martin, M. Scandolo, M. Fruchart, E. M. Munro, V. Vitelli.


The lattice simulations (in `./DIM` and `./wavepinning`) use a tau-leaping scheme implemented in `julia` to simulate microscopic particle dynamics. The `julia` package manager can run the environement in `aim1d`.
The simulations in `LandauGinzburg` use a simple Euler-Mayurama scheme to integrate the Landa-Ginzburg partial differential equations.

The notebook in `./IsingPotts/` uses `python` to solve the transfer matrix formulation of the marginal probabilities of Ising and Potts models. It only requires `numpy`,`scipy`, `tqdm` and `matplotlib` to run.
