# Safe screening rules for the identification of zeros in the solution of the Slope problem

This repository contains numerical procedure to evaluate the solution of the SLOPE problem with / without safe screening [1].
More precisely, the SLOPE problem is defined as

```math
\text{Find }
\mathbf{x}^\star\in\underset{\mathbf{x}\in\mathbb{R}^n}{\arg\min}\;
\frac{1}{2}\left\Vert 
	\mathbf{y} - \mathbf{A}\mathbf{x}
\right\Vert_2^2
+
\lambda\sum_{k=1}^n \gamma_k\vert\mathbf{x}\vert_{[k]}
```
where
 - $`\mathbf{x}\in\mathbb{R}^m`$ is the observation vector
 - $`\mathbf{A}\in\mathbb{R}^{m\times n}`$ is the so-called dictionary
 - $`\lambda>0`$ a (positive) scalar
 - $`\{\gamma_k\}_{k=1}^n`$ a sequence of non-increasing nonnegative scalars such that $`\gamma_1=1`$,
 - $`\vert\mathbf{x}\vert_{[k]}`$ denotes the $`k`$th largest entry of $`\mathbf{x}`$ in absolute value

> [1] Clément Elvira, Cédric Herzet: “Safe rules for the identification of zeros in the solution of the Slope problem”, arXiv, septembre 2021; [arXiv:1911.07508](http://arxiv.org/abs/0000.00000)

The above paper contains theoretical results and several applications that can be reproduced with this toolbox.

This python toolbox is currently under development and is hosted on Gitlab. If you encounter a bug or something unexpected please let me know by [raising an issue](https://gitlab-research.centralesupelec.fr/2020elvirac/slope-screening/-/issues) on the project page or by contacting me by mail.

## Requirements

This toolbox works with python 3.5+.

Dependencies:
 -   [NumPy](http://www.numpy.org)
 -   [SciPy](https://www.scipy.org)
 -   [Matplotlib](http://matplotlib.org)


## Install from sources

1. Clone the repository
```bash
git clone https://gitlab-research.centralesupelec.fr/2020elvirac/slope-screening
```

2. Enter the folder
```bash
cd slope-squeezing
```

3. (Optional) Create a virtual environment and activate it
```bash
virtualenv venv -p python3
source venv/bin/activate
```

4. Install the dependencies
```bash
pip install -r requirements.txt
```

5. And execute `setup.py`
```bash
pip install .
```
or 
```bash
pip install -e .
```
if you want it editable.

## Running the experiments

To reproduce the two experiments from the paper, one can simply run the launcher in the corresponding folder:
```bash
cd experiments/SIAM/exp_name
chmod a+x launcher.sh # make sure that the script is executable
./launcher.sh
```
where `exp_name` is either `xp_1_balls` or `xp_2_bench_time`.


## Licence

This software is distributed under the [CeCILL Free Software Licence Agreement](http://www.cecill.info/licences/Licence_CeCILL_V2-en.html)


## Cite this work

If you use this package for your own work, please consider citing it with this piece of BibTeX:

```bibtex
@article{Elvira2023,
	author = {Elvira, Cl\'{e}ment and Herzet, C\'{e}dric},
	title = {Safe Rules for the Identification of Zeros in the Solutions of the SLOPE Problem},
	journal = {SIAM Journal on Mathematics of Data Science},
	volume = {5},
	number = {1},
	pages = {147-173},
	year = {2023},
	doi = {10.1137/21M1457631},
	URL = {https://doi.org/10.1137/21M1457631},
}

```