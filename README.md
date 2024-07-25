# Competition on LLM designed EA

We are launching a competition centered on the innovative use of large language models (LLMs) to design evolutionary algorithms (EAs). This contest aims to explore the potential of LLMs in creating sophisticated EAs that can tackle complex optimization problems. Join us in contributing to this emerging field, showcasing how LLMs can enhance and accelerate the development of evolutionary algorithms.

The first edition of the competition will be held at the [IEEE SSCI 2025](https://cis.ieee.org/news/521-2025-ieee-symposium-series-on-computational-intelligence-ssci) (Pending acceptance).

## Benchmark definition

We will be using GNBG benchmark for box-constrained numerical global optimization [A. H. Gandomi, D. Yazdani, M. N. Omidvar, and K. Deb, "GNBG-Generated Test Suite for Box-Constrained Numerical Global Optimization," arXiv preprint arXiv:2312.07034, 2023](https://arxiv.org/abs/2312.07034).
There are 24 test functions of varying dimensions and varying problem landscapes.

## How to participate

Submit your results in the following [form](). You have to submit the following items:
* Results in the format specified below
* The used LLM with specified settings (e.g. *OpenAI GPT 4 turbo, temperature = 0.8*)
* Algorithm code (for validity check)
* Generating prompts (how did you arrive to the solution)

## Results format

You should run **31** independent runs of each test function with a stopping criterion defined by either maximum number of function evaluations (500,000 for f1-f15 and 1,000,000 for f16-f24) or reaching the acceptance threshold set to an absolute error (difference between best-found and optimum value) = 10^-8.

The results will be reported for each function by a tuple of text files - *f_x_value.txt* and *f_x_param.txt* (where x denotes the function number).
The format of the *f_x_value.txt* should be:
* 31 best-found values - one for each run on a separate line
* example in [f_x_value.txt](f_x_value.txt)

The format of the *f_x_params.txt* should be:
* 31 parameter vectors of best-found solutions (dimensions separated by a comma ,) - one for each run on a separate line
* example for a 5D problem in [f_x_params.txt](f_x_params.txt)

## Evaluation criteria

The algorithms will be ranked on the basis of their mean value on each function.
* The mean will be computed from all 31 runs
* The score range on each function is 0 to 1 (worst to best)
* The scores will be proportional
* Maximum score is therefore 24 for the algorithm that would perform the best on each benchmark function
* Maximum considered difference between algorithm performance is 10^-8
