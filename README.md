# Randomly Pivoted Cholesky

This contains code to reproduce the numerical experiments for [_Randomly pivoted Cholesky: Practical approximation of a kernel matrix with few entry evaluations_](https://arxiv.org/abs/2207.06503) by [Yifan Chen](https://yifanc96.github.io), [Ethan N. Epperly](https://www.ethanepperly.com), [Joel A. Tropp](https://tropp.caltech.edu), and [Robert J. Webber](https://rwebber.people.caltech.edu).

## Citing this Repository

If you use our code in your work, we recommend the citing the following BibTeX entry:

```bibtex
@article{chen2023randomly,
  title={Randomly pivoted Cholesky: Practical approximation of a kernel matrix with few entry evaluations},
  author={Yifan Chen and Ethan N. Epperly and Joel A. Tropp and Robert J. Webber},
  journal={arXiv preprint arXiv:2207.06503},
  year={2023}
}
```

## Algorithm

Randomly pivoted Cholesky (RPCholesky) computes a low-rank approximation to a positive semidefinite matrix $\boldsymbol{A}$.
We anticipate the algorithm is most useful for [kernel](https://en.wikipedia.org/wiki/Kernel_method) and [Gaussian process](https://en.wikipedia.org/wiki/Kriging) methods, where the matrix $\boldsymbol{A}$ is defined only implicitly by a [_kernel function_](https://en.wikipedia.org/wiki/Positive-definite_kernel) which must be evaluated at great expense to read each entry of $\boldsymbol{A}$.
The result of the algorithm is a rank $k$ [(column) Nyström approximation](https://en.wikipedia.org/wiki/Low-rank_matrix_approximations#Nyström_approximation) $\boldsymbol{\hat{A}} = \boldsymbol{F}\boldsymbol{F}^*$ to the matrix $\boldsymbol{A}$, computed in $\mathcal{O}(k^2N)$ operations and only $(k+1)N$ entry evaluations of the matrix $\boldsymbol{A}$.
In our experience, RPCholesky consistently provides approximations of comparable accuracy to [other](https://proceedings.neurips.cc/paper/2017/hash/a03fa30821986dff10fc66647c84c9c3-Abstract.html) [methods](https://jmlr.org/papers/v20/19-179.html) with fewer entry evaluations.

## Using RPCholesky

While the main purpose of these scripts is for scientific reproducibility, they also may be useful for using RPCholesky in an application.
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
F      = nystrom_approximation.F                    # Nystrom approximation is F @ F.T
pivots = nystrom_approximation.idx 
rows   = nystrom.rows                               # rows = A[pivots, :]
```

## Running the experiments

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
