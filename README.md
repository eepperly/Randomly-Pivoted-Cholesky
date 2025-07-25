# Randomly Pivoted Cholesky

This contains code to reproduce the numerical experiments for [_Randomly pivoted Cholesky: Practical approximation of a kernel matrix with few entry evaluations_](https://doi.org/10.1002/cpa.22234) by [Yifan Chen](https://yifanc96.github.io), [Ethan N. Epperly](https://www.ethanepperly.com), [Joel A. Tropp](https://tropp.caltech.edu), and [Robert J. Webber](https://rwebber.people.caltech.edu) and [_Embrace rejection: Kernel matrix approximation by accelerated randomly pivoted Cholesky_](https://arxiv.org/abs/2410.03969) by [Ethan N. Epperly](https://www.ethanepperly.com), [Joel A. Tropp](https://tropp.caltech.edu), and [Robert J. Webber](https://rwebber.people.caltech.edu).

## Citing this Repository

If you use our code in your work, please cite the following BibTeX entries:

```bibtex
@article{chen2025randomly,
  title = {Randomly Pivoted {Cholesky}: {Practical} Approximation of a Kernel Matrix with Few Entry Evaluations},
  author = {Chen, Yifan and Epperly, Ethan N. and Tropp, Joel A. and Webber, Robert J.},
  year = {2025},
  journal = {Communications on Pure and Applied Mathematics},
  volume = {78},
  number = {5},
  pages = {995--1041},
  issn = {1097-0312},
  doi = {10.1002/cpa.22234},
}


@article{epperly2024embrace,
  title={Embrace rejection: {Kernel} matrix approximation by accelerated randomly pivoted {Cholesky}},
  author={Ethan N. Epperly and Joel A. Tropp and Robert J. Webber},
  journal={Manuscript in preparation},
  year={2024}
}
```

## The RPCholesky Algorithm

Randomly pivoted Cholesky (RPCholesky) is a fast, randomized algorithm for computing a low-rank approximation to a positive semidefinite matrix $\boldsymbol{A}$ (i.e., a symmetric matrix with nonnegative eigenvalues).
We anticipate the algorithm is most useful for [kernel](https://en.wikipedia.org/wiki/Kernel_method) and [Gaussian process](https://en.wikipedia.org/wiki/Kriging) methods, where the matrix $\boldsymbol{A}$ is defined only implicitly by a [_kernel function_](https://en.wikipedia.org/wiki/Positive-definite_kernel) which must be evaluated at great expense to read each entry of $\boldsymbol{A}$.
The result of the algorithm is a rank $k$ [(column) Nyström approximation](https://en.wikipedia.org/wiki/Low-rank_matrix_approximations#Nyström_approximation) $\boldsymbol{\hat{A}} = \boldsymbol{F}\boldsymbol{F}^*$ to the matrix $\boldsymbol{A}$, computed in $\mathcal{O}(k^2N)$ operations and only $(k+1)N$ entry evaluations of the matrix $\boldsymbol{A}$.
In our experience, RPCholesky consistently provides approximations of comparable accuracy to [other](https://proceedings.neurips.cc/paper/2017/hash/a03fa30821986dff10fc66647c84c9c3-Abstract.html) [methods](https://jmlr.org/papers/v20/19-179.html) with fewer entry evaluations.
By default, this paper uses the faster _accelerated RPCholesky_ method, developed in followup work.
Accelerated RPCholesky method produces the exact same random output as ordinary RPCholesky, but is faster.

## Using RPCholesky

RPCholesky is implemented in the `rpcholesky` method in `rpcholesky.py`, and can be called as

```
nystrom_approximation = rpcholesky(A, num_pivots)
```

The input matrix `A` should be an `AbstractPSDMatrix` object, defined in `matrix.py`.
A psd matrix stored as a `numpy` array can be made usable by `rpcholesky` by wrapping it in a `PSDMatrix` object:

```
nystrom_approximation = rpcholesky(PSDMatrix(ordinary_numpy_array), num_pivots)
```

The output of `rpcholesky` is a `PSDLowRank` object (defined in `lra.py`).
From this object, one can obtain $\boldsymbol{F}$ (defining the Nyström approximation $\boldsymbol{\hat{A}} = \boldsymbol{FF}^*$), the pivot set $\mathsf{S}$, and the rows defining the Nyström approximation $A(\mathsf{S},:)$:

```
nystrom_approximation = rpcholesky(A, num_pivots)
F      = nystrom_approximation.get_left_factor()    # Nystrom approximation is F @ F.T
pivots = nystrom_approximation.get_indices() 
rows   = nystrom_approximation.get_rows()           # rows = A[pivots, :]
```

## Simple, block, and accelerated RPCholesky

This repository contains implementations of three versions of the RPCholesky algorithm.

1. Our recommended algorithm is accelerated RPCholesky. This algorithm produces the same random distribution of outputs as simple RPCholesky, but uses a faster, rejection-sampling based design. This is our default RPCholesky algorithm. To compute a rank-$k$ approximation, one can use the invocation:
```python
nystrom_approximation = rpcholesky(A, k)
```
The accelerated RPCholesky algorithm has an optional block size option, which can be set as follows:
```python
nystrom_approximation = rpcholesky(A, k, b = block_size)
```
The accelerated RPCholesky method can also be called using the dedicated function `accelerated_rpcholesky`.

2. Block RPCholesky is anothter faster, blocked RPCholesky algorithm. We generally recommend accelerated RPCholesky over it, as block RPCholesky can produce less accurate approximations on certain problems. Block RPCholesky can be invoked using the `block_rpcholesky` function or using `rpcholesky` with a specified block size and the `accelerated` flag set to `False`:
```python
nystrom_approximation = rpcholesky(A, k, b = block_size, accelerated = False)
```

3. Simple RPCholesky is the basic, unaccelerated version of RPCholesky. We generally recommend against its use because it is slower than accelerated RPCholesky. It can be invoked using the `simple_rpcholesky` function or by calling `rpcholesky` with the `accelerated` flag set to `False`:
```python
nystrom_approximation = rpcholesky(A, k, accelerated = False)
```

## Running the experiments for the accelerated RPCholesky paper

The first step to reproducing the experiments from the manuscript is to run the script

```
./setup.sh
```

which sets up the file structure.
The experiments for the accelerated RPCholesky paper are found in the folder `block_experiments/`.
The relevant files are described below:

1. `block_experiments/block_generation.py`: tests the runtime of generating columns of Gaussian and Laplace kernel matrices for different block sizes. Used to produce Figure 1.
2. `block_experiments/test_accelerated.py`: provides an initial comparison of accelerated, block, and simple RPCholesky on a synthetic matrix. Used to produce Figure 2.
3. `block_experiments/performance.py`: tests accelerated, block, and simple RPCholesky on a large bed of examples. Used to produce Figure 3.
4. `block_experiments/pes_code.py`: evaluates accelerated RPCholesky and alternatives for computation of potential energy surfaces. Used to produce the experiments in section 3.2.
5. `block_experiments/compare_rbrp.py`: compares accelerated RPCholesky, block RPCholesky, and RBRP Cholesky. Used to form the table in the appendix.

Once the relevant Python scripts have been run, the figures from the paper can be generated from the relevant MATLAB scripts in `experiments/matlab_plotting/`.

## Running the experiments for the original RPCholesky paper

The first step to reproducing the experiments from the manuscript is to run the script

```
./setup.sh
```

which sets up the file structure, loads RLS and DPP samplers, and downloads the QM9 dataset for the KRR example.
The data from the figures in the paper can produced by running the following scripts in the `experiments/` folder, each of which has instructions for its individual use at a comment at the top:

1. `experiments/comparison.py`: compares the approximation error for different Nyström methods. Used to produce the left displays in Figure 1.
2. `experiments/chosen.py`: outputs the pivots chosen by different Nyström methods. Used to produce the right displays in Figure 1.
3. `experiments/entries.py`: outputs the entry evaluations for different Nyström methods. Used to produce Figure 2.
4. `experiments/qm9_krr.py`: performs kernel ridge regression on the QM9 dataset. Used to produce Figure 3.
5. `experiments/cluster_biomolecule.py`: performs spectral clustering on the [alanine dipeptide dataset](https://markovmodel.github.io/mdshare/ALA2/). Used to produce Figure 4.
6. `experiments/timing.py`: compares the timing of different Nyström methods.

Once the relevant Python scripts have been run, the figures from the paper can be generated from the relevant MATLAB scripts in `experiments/matlab_plotting/`.

Figure 4 in the manuscript was completely changed in revision. Figure 4 from [earlier versions of the manuscript](https://arxiv.org/abs/2207.06503v3) can be generated using the scripts `experiments/cluster_letters.py` and `experiments/cluster_letters_plot.py`.
