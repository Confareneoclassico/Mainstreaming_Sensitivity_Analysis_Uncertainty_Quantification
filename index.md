
# Mainstreaming Sensitivity Analysis and Uncertainty Auditing

<!-- AUTHOR: Samuele lo Piano -->
<!-- AUTHOR: Leif Rune Hellevik -->
<!-- AUTHOR: Andrea Saltelli -->
<!-- AUTHOR: Philip B. Stark  -->
<!-- AUTHOR: Jeroen Van der Sluijs -->

**Samuele Lo Piano**, **Leif Rune Hellevik**, **Andrea Saltelli**, **Philip B. Stark**, **Jeroen Van der Sluijs**

In this Git repository we have collected a number of notebooks that introduce uncertainty quantification and sensitivity analysis (UQSA). The material covers both epistemic (unknown and unquantifiable) uncertainty as well as stochastic uncertainty. This latter can be apportioned to the model input parameters through sensitivity analysis. Global sensitivity analysis - the approach where all the parameters are varied at the same time - is presented in its variance-based form through the calculation of Sobol sensitivity indices. The use of meta-modelling for sensitivity analysis is also described through Polynomial Chaos. Finally, sensitivity auditing - an approach to check the normative frames of any modelling activity undertaken - is also described.

 The notebooks on the above-mentioned sub-topics are found in the following sub-sections.

## Uncertainty and quality in science for policy

[In this notebook](WebResources.ipynb), the reader can find useful web-resources on this topic. A Python-based [interactive tool for the visualization of the NUSAP (Numeral Unit Spread Assessment Pedigree) approach](Interactive_plot.ipynb) to visualise NUSAP experts scores across categories is also part of this collection.

* [Resources on _Uncertainty and quality in science for policy_](WebResources.ipynb)

* [Visualisation tools for pedigree scores](Interactive_plot.ipynb)

## Uncertainty quantification and sensitivity analysis tutorials

Sensitivity analysis starting concepts can be found in the [book of Saltelli et al 2008](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184) and in [this collection of notebooks](https://github.com/lrhgit/uqsa_tutorials)along with applications to the fields of biomechanics. Statistical preliminaries can be found [here](https://github.com/lrhgit/uqsa_tutorials/blob/master/preliminaries.ipynb) along with a [practical introduction](https://github.com/lrhgit/uqsa_tutorials/blob/master/sensitivity_introduction.ipynb), the [comparison of the global and local approach](https://github.com/lrhgit/uqsa_tutorials/blob/master/local_vs_global.ipynb) and the computations required to calculate [high order indices](https://github.com/lrhgit/uqsa_tutorials/blob/master/sensitivity_higher_order.ipynb). The features of the [Monte Carlo approach](https://github.com/lrhgit/uqsa_tutorials/blob/master/monte_carlo.ipynb) along with the [polynomial chaos expansion](https://github.com/lrhgit/uqsa_tutorials/blob/master/introduction_gpc.ipynb) are then presented along with an application to the [Ishigami test function](https://github.com/lrhgit/uqsa_tutorials/blob/master/ishigami_example.ipynb). [An application to the field of biomechanics](https://github.com/lrhgit/uqsa_tutorials/blob/master/wall_models.ipynb) along with [interactive exercises](https://github.com/lrhgit/uqsa_tutorials/blob/master/exercises.ipynb) are the final components of this collection of notebooks.

### New notebooks on sensitivity analysis

 The notebooks developed in this series aim at presenting further tools useful to students, modelers and practitioners from the quantitative-assessment field. Specifically, the use of the [Sobol sequence](sobol_interactive.ipynb) -or other low-discrepancy sequences- can be fundamental in case of computationally demanding models. This feature can be appreciated by testing the convergence pace of the Sobol sequence against a purely random approach with a [series of test functions](testfunctions3.ipynb), that reproduce additive, non-additive and higher-order-interaction models. The so-called [G and G* test functions](https://www.sciencedirect.com/science/article/pii/S0010465509003087) can help to clarify this distinction by tuning the value of the additive constant in the model. The same base functions are typically used by practitioners from the field to [benchmark the performance](interactive_gstar_function.ipynb) of different approaches as illustrated for the [variance-based estimator](https://www.sciencedirect.com/science/article/pii/S0010465509003087) against [the polynomial chaos approach](interactive_g_function.ipynb). Finally, an application to a real case is presented by assessing whether [silver shortage could constrain large scale development of solar photovoltaics](https://github.com/pbstark/SA/blob/master/New_notebooks/silver2.ipynb) is presented.

* [Sobol sequence](sobol_interactive.ipynb)

* [Test functions for testing convergence pace](testfunctions3.ipynb)

* [Introduction to polynomial chaos with chaospy](introduction_gpc.ipynb)

* [Comparison of the polynomial chaos and variance-based estimator approaches for the G function](interactive_g_function.ipynb)

* [Comparison of the polynomial chaos and variance-based estimator approaches for the G* function](interactive_gstar_function.ipynb)

* [Comparison of the polyomial chaos and variance-based estimator approaches for the other test functions](PC_test_functions.ipynb)

* [Silver as a potential constraint to large-scale development of photovoltaics](silver2.ipynb)

## Sensitivity auditing

This approach is an enhancement of sensitivity analysis to the full normative aspects of the modelling activity. A thorough description of the approach with a series of examples can be found [here](sensitivity_auditing.ipynb).

* [Sensitivity Auditing](sensitivity_auditing.ipynb)

## Acknowledgements

This collection of notebooks were developed with financial support from the
[Peder Sather Center for Advanced Study](http://sathercenter.berkeley.edu)