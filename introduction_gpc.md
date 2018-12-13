<!-- dom:TITLE: A practical introduction to polynomial chaos with the chaospy
package -->
# A practical introduction to polynomial chaos with the chaospy package
<!-- dom:AUTHOR: Vinzenz Gregor Eck at Expert Analytics, Oslo -->
<!--
Author: -->  
**Vinzenz Gregor Eck**, Expert Analytics, Oslo  
<!-- dom:AUTHOR:
Jacob Sturdy at Department of Structural Engineering, NTNU -->
<!-- Author: -->
**Jacob Sturdy**, Department of Structural Engineering, NTNU  
<!-- dom:AUTHOR:
Leif Rune Hellevik at Department of Structural Engineering, NTNU -->
<!-- Author: --> 
**Leif Rune Hellevik**, Department of Structural Engineering, NTNU
Date: **Sep 25, 2018**

```{.python .input}
# ipython magic
%matplotlib notebook
%load_ext autoreload
%autoreload 2
```

```{.python .input}
# plot configuration
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# import seaborn as sns # sets another style
matplotlib.rcParams['lines.linewidth'] = 3
fig_width, fig_height = (7.0,5.0)
matplotlib.rcParams['figure.figsize'] = (fig_width, fig_height)
```

```{.python .input}
import chaospy as cp
import numpy as np
from numpy import linalg as LA
```

# Introduction
<div id="sec:introduction"></div>

To conduct UQSA with the
polynomial chaos (PC) method we use the python package `chaospy`.
This package
includes many features and methods to apply non-intrusive polynomial chaos to
any model with
few lines of code.
Since, `chaospy` allows for convenient
definition of random variables, calculation of joint distributions and
generation of samples,
we apply `chaospy` for the Monte Carlo method as well.
In this notebook we will briefly describe the theory of polynomial chaos and
give a practical step by step guide
for the application of `chaospy`.

# Polynomial chaos <div id="sec:polynomialChaos"></div>

For a more in depth
introduction to the polynomial chaos method see [Xiu 2010](https://press.princeton.edu/titles/9229.html) or [Smith 2013](http://bookstore.siam.org/cs12/)

## Concept

The main concept of polynomial chaos is to approximate the stochastic
model response of a model with stochastic input as polynomial expansion relating
the response to the inputs.
For simplicity, we consider a model function $f$
which takes random variables $\mathbf{Z}$ and non-random variables as
spatial
coordinates $x$ or time $t$, as input:

<!-- Equation labels as ordinary links -->
<div id="_auto1"></div>

$$
\begin{equation}
Y = y{}(x, t, \mathbf{Z})
\label{_auto1} \tag{1}
\end{equation}
$$

Then we seek to generate a polynomial expansion such that one can approximate the
model response.

<!-- Equation labels as ordinary links -->
<div id="_auto2"></div>

$$
\begin{equation}
Y = y{}(x, t, \mathbf{Z}) \approx y_{N} = \sum_{i=1}^{P} v_i(x,
t) \Phi_i(\mathbf{Z}),
\label{_auto2} \tag{2}
\end{equation}
$$

where $v_i$ are the expansion coefficients, which are only dependent on the non-
stochastic parameters, and $\Phi_i$ are orthogonal polynomials, which depend
solely on the random input parameter $\mathbf{Z}$.
Once the polynomial expansion
is calculated, we can analytically calculate the statistics of approximated
model output.

Since polynomial chaos generally requires fewer model evaluations
to estimates statistics than Monte Carlo methods, this method is preferable
applied to computational-demanding models.

### Applications
Polynomial chaos may be applied in a wide variety of situations. Requirements
for convergence are simply that the function be square integrable with respect
to the inner product associated with the orthogonal polynomials. In other words
if the function has a finite variance, polynomial chaos may be applied without
worrying too much. The caveat to this fact is that convergence may not
necessarily be fast: The smoother the function, the faster the convergence.
Polynomial chaos can handle discontinuities, however, it may be advisable to
reformulate the problem in these situations (e.g. see [Feinberg et al. 2015](https://www.sciencedirect.com/science/article/pii/S1877750315300119)).

## Orthogonal polynomials

As stated above, the polynomial chaos
expansion consists of a sum of basis polynomials in the input parameters
$\mathbf{Z}$. By using orthogonal polynomials as the basis polynomials, the
efficiency of the convergence may be improved and the use of the expansion in
uncertainty quantification and sensitivity analysis is simplified.
Orthogonality of functions is a general concept developed for functional analysis within an
inner product space. Typically the inner product of two functions is defined as
a weighted integral of the product of the functions over a domain of interest. 
<hr/>
In particular, the inner product of
two polynomials is defined as the expected value of their product for the purposes of polynomial chaos. I.e. the
integral of their product weighted with respect to the distribution of random
input parameters.
Two polynomials are orthogonal if their inner product is 0,
and a set of orthogonal polynomials is orthogonal if every pair of distinct
polynomials are orthogonal.

The following equalities hold for the orthogonal
basis polynomials used in  polynomial chaos

$$
\begin{align*}
 \mathbb{E}(\Phi_i(\mathbf{Z})\Phi_j(\mathbf{Z})) &= \langle
\Phi_i(\mathbf{Z}),\Phi_j(\mathbf{Z}) \rangle\\
 &= \int_{\Omega}
\Phi_i(z)\Phi_j(z)w(z) dz \\
 &= \int_{\Omega} \Phi_i(z)\Phi_j(z)f(z) dz  \\
 &=
{\int_{\Omega} \Phi_i(z)\Phi_j(z) dF_Z(\mathbf{z})} = h_j \delta_{ij},
\end{align*}
$$

where $h_j$ is equal to the normalisation factor of the used polynomials, and
$\Phi_i(\mathbf{Z})$ indicates the substitution of the random variable
$\mathbf{Z}$ as the polynomial's variable.
Note that $\Phi_0$ is a polynomial of
degree zero, i.e. a constant thus $\mathbb{E}(\Phi_0\Phi_j) = \Phi_0
\mathbb{E}(\Phi_j) = 0$ which implies that $\mathbb{E}(\Phi_j) = 0$ for $j \neq
0$.
<hr/>
Once the random inputs $\mathbf{Z}$ are properly defined with
marginal distributions, orthogonal polynomials can be constructed.
For most univariate distributions, polynomial bases functions are defined, and listed in
the Asker Wilkeys scheme.
A set of orthogonal polynomials can be created from
those basis polynomials with orthogonalization methods as Cholesky
decomposition, three terms recursion, and modified Gram-Schmidt.

## Expansion coefficients

There are two non-intrusive ways of approximating a polynomial
chaos expansion coefficients:

### Regression

Supposing a polynomial expansion
approximation $y_{N} = \sum_i v_i \Phi_i$,
then *stochastic collocation*
specifies a set of nodes, $\Theta_{M} =
\left\{\mathbf{z}^{(s)}\right\}_{s=1}^{P}$, where the deterministic values of
$y_{N}=y$.
The task is thus to find suitable coefficients $v_i$ such that this
condition is satisfied.
Considering the existence of a set of collocation nodes
$\left\{\mathbf{z}^{(s)}\right\}_{s=1}^{P}$, then $y_{N} = \sum_i v_i \Phi_i$
can be
 formed into linear system of equations for the coefficients $v_i$ at
these nodes:

<!-- Equation labels as ordinary links -->
<div id="eq:stochColl"></div>

$$
\begin{equation}
\begin{bmatrix}
\Phi_0(\mathbf{z}^{(1)}) & \cdots &
\Phi_P(\mathbf{z}^{(1)}) \\
\vdots & & \vdots \\
\Phi_0(\mathbf{z}^{(N)}) &
\cdot & \Phi_P(\mathbf{z}^{(n)})
\end{bmatrix}
\begin{bmatrix}
v_{0}\\
\vdots \\
v_{P}
\end{bmatrix}
= \begin{bmatrix}
y(\mathbf{z}^{(1)}) \\
\vdots \\
y
(\mathbf{z}^{(N)})
\end{bmatrix}
\end{equation}
\label{eq:stochColl} \tag{3}
$$

Now we can use regression to achieve the relaxed condition that $y_{N}$ is
"sufficiently close" to $y$ at $\left\{\mathbf{z}^{(s)}\right\}_{s=1}^{P}$.
This
is done by choosing a larger number of samples so that ([3](#eq:stochColl)) is
over determined and
then minimizing the appropriate error ${\lvert\lvert {y_{N}
\rvert\rvert}-y}_{R}$ over $\left\{\mathbf{z}^{(s)}\right\}_{s=1}^{P}$.
Ordinary
least squares, Ridge-Regression, and Tikhonov-Regularization are all regression
methods that may be applied to this problem.

### Pseudo-spectral projection

Discrete projection refers to the approximation of coefficients for $y_{N} =
\sum_{i=1}^{P} v_i \Phi_i$, by directly approximating the orthogonal projection
coefficients

<!-- Equation labels as ordinary links -->
<div id="_auto3"></div>

$$
\begin{equation}
v_i = \frac{1}{h_i} \mathbb{E}(y \Phi_i) = \frac{1}{h_i}
\int_{\Omega} y \Phi_i dF_z,
\label{_auto3} \tag{4}
\end{equation}
$$

using a quadrature scheme to calculate the integral $\int_{\Omega} y \Phi_i
dF_z$ as a sum $\sum_{i=1}^{P} w_i y(\mathbf{z}^{(i)})$, where $w_i$ are the
quadrature weights.

This results in an approximation $\tilde{v}_{i}$ of $v_{i}$
the error between the final approximation may be split

<!-- Equation labels as ordinary links -->
<div id="eq:ps_error"></div>

$$
\begin{equation}
\lVert{y_{N} - y}\rVert \leq \lVert \sum_{i=1}^{P} v_i \Phi_i -
y{} \rVert + \lVert {\sum_{i=1}^{P} \left( v_i - \tilde{v}_i \right)\Phi_i}
\rVert
\label{eq:ps_error} \tag{5}
\end{equation}
$$

where the first term is called the truncation error and the second term the
quadrature error.
Thus one may consider the maximal accuracy for a given
polynomial order $P$ and see this should be achieved as the quadrature error is
reduced to almost 0 by increased number of collocation nodes.

## Statistics

Once the expansion is generated, it can be used directly to calculated
statistics for the uncertainty and sensitivity analysis.
The two most common
measures for uncertainty quantification, **expected value** and **variance**,
can be calculated by
inserting the expansion into the definition of the
measures.

The expected value is equal to the first expansion coefficient:

<!-- Equation labels as ordinary links -->
<div
id="eq:expectedValue_gPCE"></div>

$$
\begin{equation}
  \begin{aligned}
{\mathbb{E}}(Y) \approx \int_{\Omega} \sum_{i=1}^{P} v_i \Phi_i(\mathbf{z}{})
dF_Z(\mathbf{z}) = v_1.
  \end{aligned}
    \label{eq:expectedValue_gPCE}
\tag{6}
\end{equation}
$$

The variance is the sum of squared expansion coefficients multiplied by
normalisation constants of the polynomials:

<!-- Equation labels as ordinary links -->
<div id="eq:variance_gPCE"></div>

$$
\begin{equation}
    \begin{aligned}
    \operatorname{Var}(Y) &\approx
{\mathbb{E}}{({v(x,t,\mathbf{Z})}(\mathbf{Z})-{\mathbb{E}}(Y))^2} =
\int_{\Omega}({v(x,t,\mathbf{Z})}(\mathbf{z}) - v_1)^2 dF_Z(\mathbf{z}) \\
&=  \int_{\Omega}\left(\sum_{i=1}^{P} v_i \Phi_i(\mathbf{z}) \right)^2
dF_Z(\mathbf{z}) - v_1^2 = \sum_{i=1}^{P} v_i^2
\int_{\Omega}{\Phi^2_i(\mathbf{z})}dF_Z(\mathbf{z}) - v_1^2 \\
    &=
\sum_{i=1}^{P} v_i^2 h_i - v_1^2 = \sum_{i=2}^{P} v_i^2 h_i
  \end{aligned}
\label{eq:variance_gPCE} \tag{7}
\end{equation}
$$

(Note the orthogonality of individual terms implies their covariance is zero,
thus the variance is simply the sum of the variances of the terms.)

The Sobol
indices may be calculated quite simply from the expansion terms due to the fact
that the _ANOVA_ decomposition is unique. Thus, the main effect of a parameter
$z_i$ is simply the variance of all terms only in $z_i$. Explicitly let
$\mathcal{A}_{i} = \{k | \Phi_{k}(\mathbf{z}) = \Phi_{k}(z_i)\} $ ,i.e.
$\mathcal{A}_{i}$ is the set of all indices of basis functions depending only on
$z_i$ then

<!-- Equation labels as ordinary links -->
<div id="eq:sensitivity_gPCE"></div>
$$
\begin{equation}
    \begin{aligned}
    f_i &= \sum_{k\in \mathcal{A}_{i}}
v_{k} \Phi_{k} \implies \\
    S_i &= \frac{1}{\operatorname{Var}(Y)} \sum_{k\in
\mathcal{A}_{i}} \operatorname{Var}(v_{k} \Phi_{k}) \\
    S_i &=
\frac{1}{\operatorname{Var}(Y)} \sum_{k\in \mathcal{A}_{i}} v_{k}^2 h_{k}
\end{aligned}
  \label{eq:sensitivity_gPCE} \tag{8}
\end{equation}
$$

and similarly one may define $\mathcal{A}_{ij}$ for pairwise combinations of
inputs and further to calculate all orders of sensitivities.

# Chaospy <div id="sec:chaospy"></div>

The python package `chaospy` an introductory paper to
the package including a comparison to other software packages is
presented here [Feinberg 2015](https://www.sciencedirect.com/science/article/pii/S1877750315300119).
You can find an introduction, tutorials and
the source code at the [projects homepage](https://github.com/jonathf/chaospy)
The installation of the package can be done via `pip`:

        pip install chaospy

In the following we will use the import naming convention of the package creator
to import the package in python:

        import chaospy as cp

Therefore it will be convenient to see whenever a method of the package is
applied.

The package `chaospy` is doc-string annotated, which means that every
method provides a short help text with small examples.
To show the method
documentation simply type a `?` after the method name in a ipython console or
notebook.
As shown in the following two examples:

```{.python .input}
# show help for uniform distributions
cp.Uniform?
```

```{.python .input}
# show help for sample generation
cp.samplegen?
```

## Steps for polynomial chaos analysis with chaospy
<div
id="sec:uqsaChaospy"></div>

To conduct UQSA analysis with polynomial chaos we
need to follow the following steps:

* Definition of the marginal and joint distributions

* Generation of the orthogonal polynomials

* Linear regression

  * Generation of samples

  * `Evaluation of the model for all samples`

  * Generation of the polynomial chaos expansion

* Pseudo-spectral projection

  * Generation of integration nodes and weights

  * `Evaluation of the model for all nodes`

  * Generation of the polynomial chaos expansion

* Calculations of all statistics

Note, that steps  **3 Linear regression**  and **4 Pseudo-spectral projection** are interchangeable. They are simply different methods of
calculating the expansion coefficients. In both cases
one generates a set of points in the parameter space where the model must be evaluated (steps 3.b and 4.b,
respectively).


## Step 1: Definition of marginal and joint distributions
<div
id="sec:distributions"></div>

The analysis of a each model starts with the
definition of the marginal distributions for each random model input, i.e.
describing it as
random variable.
Univariate random variables can be defined
with `chaospy` by calling the class-constructor of a distribution type, e.g
`cp.Normal()`, with arguments to describe the particular distribution, e.g. mean
value and standard deviation for `cp.Normal`.
The help function can be used to
find out more about the required arguments, e.g. `help(cp.Normal)`.

In the
following an example for 3 random variables with uniform, normal and log-normal
distribution:

```{.python .input}
# simple distributions
rv1 = cp.Uniform(0, 1)
rv2 = cp.Normal(0, 1)
rv3 = cp.Lognormal(0, 1, 0.2, 0.8)
print(rv1, rv2, rv3)
```

After all random input variables are defined with univariate random variables a
multi-variate random variable and its joint distribution
can be constructed with
the following command:

```{.python .input}
# joint distributions
joint_distribution = cp.J(rv1, rv2, rv3)
print(joint_distribution)
```

It is also possible to construct independent identical distributed random
variables from any univariate variable:

```{.python .input}
# creating iid variables
X = cp.Normal()
Y = cp.Iid(X, 4)
print(Y)
```

All 64 distributions available in the chaospy package can be found in the
following table:

<table border="1">
<thead>
<tr><th align="center">
Distributions   </th> <th align="center">     implemented     </th> <th
align="center">       in chaospy      </th> <th align="center">
</th> </tr>
</thead>
<tbody>
<tr><td align="center">   Alpha
</td> <td align="center">   Birnbaum-Sanders         </td> <td align="center">
Laplace                    </td> <td align="center">   Power log-normal
</td> </tr>
<tr><td align="center">   Anglit                 </td> <td
align="center">   Fisher-Snedecor          </td> <td align="center">   Levy
</td> <td align="center">   Power normal             </td> </tr>
<tr><td
align="center">   Arcsinus               </td> <td align="center">   Fisk/log-
logistic        </td> <td align="center">   Log-gamma                  </td> <td
align="center">   Raised cosine            </td> </tr>
<tr><td align="center">
Beta                   </td> <td align="center">   Folded Cauchy
</td> <td align="center">   Log-laplace                </td> <td align="center">
Rayleigh                 </td> </tr>
<tr><td align="center">   Brandford
</td> <td align="center">   Folded normal            </td> <td align="center">
Log-normal                 </td> <td align="center">   Reciprocal
</td> </tr>
<tr><td align="center">   Burr                   </td> <td
align="center">   Frechet                  </td> <td align="center">   Log-
uniform                </td> <td align="center">   Right-skewed Gumbel
</td> </tr>
<tr><td align="center">   Cauchy                 </td> <td
align="center">   Gamma                    </td> <td align="center">   Logistic
</td> <td align="center">   Student-T                </td> </tr>
<tr><td
align="center">   Chi                    </td> <td align="center">   Gen.
exponential         </td> <td align="center">   Lomax                      </td>
<td align="center">   Triangle                 </td> </tr>
<tr><td
align="center">   Chi-square             </td> <td align="center">   Gen.
extreme value       </td> <td align="center">   Maxwell                    </td>
<td align="center">   Truncated exponential    </td> </tr>
<tr><td
align="center">   Double Gamma           </td> <td align="center">   Gen. gamma
</td> <td align="center">   Mielke's beta-kappa        </td> <td align="center">
Truncated normal         </td> </tr>
<tr><td align="center">   Double Weibull
</td> <td align="center">   Gen. half-logistic       </td> <td align="center">
Nakagami                   </td> <td align="center">   Tukey-Lambda
</td> </tr>
<tr><td align="center">   Epanechnikov           </td> <td
align="center">   Gilbrat                  </td> <td align="center">   Non-
central chi-squared    </td> <td align="center">   Uniform
</td> </tr>
<tr><td align="center">   Erlang                 </td> <td
align="center">   Truncated Gumbel         </td> <td align="center">   Non-
central Student-T      </td> <td align="center">   Wald
</td> </tr>
<tr><td align="center">   Exponential            </td> <td
align="center">   Gumbel                   </td> <td align="center">   Non-
central F              </td> <td align="center">   Weibull
</td> </tr>
<tr><td align="center">   Exponential power      </td> <td
align="center">   Hypergeometric secant    </td> <td align="center">   Normal
</td> <td align="center">   Wigner                   </td> </tr>
<tr><td
align="center">   Exponential Weibull    </td> <td align="center">   Kumaraswamy
</td> <td align="center">   Pareto (first kind)        </td> <td align="center">
Wrapped Cauchy           </td> </tr>
</tbody>
</table>

## Step 2: Orthogonal Polynomials
<div id="sec:orthogonalPolynomials"></div>

The orthogonal
polynomials can be generated with different methods, in `chaospy` there are 4
methods implemented. The
most stable method, and therefore most advised is the
*three terms recursion* method.

<table border="1">
<thead>
<tr><th
align="center">Orthogonalization Method</th> <th align="center">
</th> </tr>
</thead>
<tbody>
<tr><td align="center">   Cholesky decomposition
</td> <td align="center">   cp.orth\_chol    </td> </tr>
<tr><td align="center">
Three terms recursion       </td> <td align="center">   cp.orth\_ttr     </td>
</tr>
<tr><td align="center">   Modified Gram-Schmidt       </td> <td
align="center">   cp.orth\_gs      </td> </tr>
</tbody>
</table>

Regarding the *three terms recursion* method:
For the distributions Normal, Uniform, Gamma,
Log-normal, Triangle, Beta and stochastic independent variable combinations of
those,
the three terms recursion coefficients are known.
For all other
distributions the coefficients are estimated numerically.
The *three terms
recursion* method is then also called **discretized stieltjes method**.

The
most stable method and therefore most applied method is the **three terms
recursion** (**discretized stieltjes method**) method.

We will look at all in a
small example, try to increase the polynomial order and the instabilities of the
methods become visible.

```{.python .input}
# example orthogonalization schemes
# a normal random variable
n = cp.Normal(0, 1)

x = np.linspace(0,1, 50)
# the polynomial order of the orthogonal polynomials
polynomial_order = 3

poly = cp.orth_chol(polynomial_order, n, normed=True)
print('Cholesky decomposition {}'.format(poly))
ax = plt.subplot(131)
ax.set_title('Cholesky decomposition')
_=plt.plot(x, poly(x).T)
_=plt.xticks([])

poly = cp.orth_ttr(polynomial_order, n, normed=True)
print('Discretized Stieltjes / Three terms reccursion {}'.format(poly))
ax = plt.subplot(132)
ax.set_title('Discretized Stieltjes ')
_=plt.plot(x, poly(x).T)

poly = cp.orth_gs(polynomial_order, n, normed=True)
print('Modified Gram-Schmidt {}'.format(poly))
ax = plt.subplot(133)
ax.set_title('Modified Gram-Schmidt')
_=plt.plot(x, poly(x).T)
```

## Step 3.: Linear regression

The linear regression method requires to conduct
the three following steps:

1. Generation of samples

2. `Evaluation of the model for all samples`

3. Generation of the polynomial chaos expansion

In the following steps, we will not consider the model evaluation.

### Step 3.a: Sampling
<div id="sec:sampling"></div>

Once a random variable is defined or a joint
random variable -referred as distribution here, the following
method can be
used to generate as set of samples:

```{.python .input}
# sampling in chaospy
u = cp.Uniform(0,1)
u.sample?
```

The method takes the arguments **size** which is the number of samples and
**rule** which is the applied sampling scheme.
The following example shows the
creation of 2 set of samples for the sampling schemes *(Pseudo-)Random* and
*Hammersley*.

```{.python .input}
# example sampling
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)
number_of_samples = 350
samples_random = joint_distribution.sample(size=number_of_samples, rule='R')
samples_hammersley = joint_distribution.sample(size=number_of_samples, rule='M')

fig1, ax1 = plt.subplots()
ax1.set_title('Random')
ax1.scatter(*samples_random)
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')

fig2, ax2 = plt.subplots()
ax2.set_title('Hammersley sampling')
ax2.scatter(*samples_hammersley)
ax2.set_xlabel("Uniform 1")
ax2.set_ylabel("Uniform 2")
ax2.axis('equal')
```

All sampling schemes implemented in chaospy are listed in the following table:
<table border="1">
<thead>
<tr><th align="center">Key</th> <th align="center">
Name      </th> <th align="center">Nested</th> </tr>
</thead>
<tbody>
<tr><td
align="center">   C      </td> <td align="center">   Chebyshev nodes     </td>
<td align="center">   no        </td> </tr>
<tr><td align="center">   NC
</td> <td align="center">   Nested Chebyshev    </td> <td align="center">   yes
</td> </tr>
<tr><td align="center">   K      </td> <td align="center">   Korobov
</td> <td align="center">   no        </td> </tr>
<tr><td align="center">   R
</td> <td align="center">   (Pseudo-)Random     </td> <td align="center">   no
</td> </tr>
<tr><td align="center">   RG     </td> <td align="center">   Regular
grid        </td> <td align="center">   no        </td> </tr>
<tr><td
align="center">   NG     </td> <td align="center">   Nested grid         </td>
<td align="center">   yes       </td> </tr>
<tr><td align="center">   L
</td> <td align="center">   Latin hypercube     </td> <td align="center">   no
</td> </tr>
<tr><td align="center">   S      </td> <td align="center">   Sobol
</td> <td align="center">   yes       </td> </tr>
<tr><td align="center">   H
</td> <td align="center">   Halton              </td> <td align="center">   yes
</td> </tr>
<tr><td align="center">   M      </td> <td align="center">
Hammersley          </td> <td align="center">   yes       </td> </tr>
</tbody>
</table>

### Step 3.a.i: Importing and exporting samples

It may be useful to export the samples from `chaospy` for use in another program. The most useful
format for exporting the samples likely depends on the external program, but it
is quite straightforward to save the samples as a `CSV` file with a delimeter of your
choice:

```{.python .input}
# example save samples to file
# Creates a csv file where each row corresponds to the sample number and each column with teh variables in the joint distribution
csv_file = "csv_samples.csv"
sep = '\t'
header = ["u1", "u2"]
header = sep.join(header)
np.savetxt(csv_file, samples_random, delimiter=sep, header=header)
```

Each row of the csv file now contains a single sample from the joint
distribution with the columns corresponding to each component.

Now you may
evaluate these samples with an external program and save the resulting data into
a similarly formatted `CSV` file. Again each row should correspond to a single
sample value and each column to different components of the model output.

```{.python .input}
# example load samples from file
# loads a csv file where the samples/or model evaluations for each sample are saved
# with one sample per row. Multiple components ofoutput can be stored as separate columns
filepath = "external_evaluations.csv"
data = np.loadtxt(filepath)
```

### Step 3.c: Polynomial Chaos Expansion

After the model is evaluated for all
samples, the polynomial chaos expansion can be generated with the following
method:

```{.python .input}
# linear regression in chaospy
cp.fit_regression?
```

A complete example for polynomial chaos expansion using
the linear regression follows.
The model applied the very simple mathematical
expression:

<!-- Equation labels as ordinary links -->
<div id="eq:dummy_model"></div>

$$
\begin{equation}
 y(z_1, z_2) = z_1 + z_1 z_2
\label{eq:dummy_model} \tag{9}
\end{equation}
$$

The random variables for $Z_1, Z_2$ are defined as simple uniform random
variables:

<!-- Equation labels as ordinary links -->
<div id="eq:dummy_rv"></div>

$$
\begin{equation}
Z_1 = \mbox{U}(0,1), \quad Z_2 = \mbox{U}(0,1)
\label{eq:dummy_rv} \tag{10}
\end{equation}
$$

The mean of this should be $\frac{3}{4}$, the variance should be
$\frac{31}{144}$ and the sensitivites to $Z_1$ and $Z_2$ are respectively
$\frac{3}{31}$ and $\frac{27}{31}$.

Here is the annotated example code with all
steps required to generate a polynomial chaos expansion with
linear regression:

```{.python .input}
# example linear regression
# 1. define marginal and joint distributions
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

# 2. generate orthogonal polynomials
polynomial_order = 3
poly = cp.orth_ttr(polynomial_order, joint_distribution)

# 3.1 generate samples
number_of_samples = cp.bertran.terms(polynomial_order, len(joint_distribution))
samples = joint_distribution.sample(size=number_of_samples, rule='R')

# 3.2 evaluate the simple model for all samples
model_evaluations = samples[0]+samples[1]*samples[0]

# 3.3 use regression to generate the polynomial chaos expansion
gpce_regression = cp.fit_regression(poly, samples, model_evaluations)
print("Success")
```

## Step 4: Pseudo-spectral projection

### Step 4.a: Quadrature nodes and weights
<div id="sec:quadrature"></div>

Once a random variable is defined or
joint random variables, also referred as distribution here, the following
method
can be used to generate nodes and weights for different quadrature methods:

```{.python .input}
# quadrature in polychaos
cp.generate_quadrature?
```

We will look at the following arguments of the method: **order** is the order of
the quadrature, **domain** is
the distribution one is working on, **rule** is the *name* or *key* of the
quadrature rule to apply.

In the following example we look at some quadrature
nodes for the same uniform variables as for the sampling,
for Optimal Gaussian
quadrature and Clenshaw-Curtis quadrature.

```{.python .input}
# example quadrature
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

order = 5

nodes_gaussian, weights_gaussian = cp.generate_quadrature(order=order, domain=joint_distribution, rule='G')
nodes_clenshaw, weights_clenshaw = cp.generate_quadrature(order=order, domain=joint_distribution, rule='C')

print('Number of nodes gaussian quadrature: {}'.format(len(nodes_gaussian[0])))
print('Number of nodes clenshaw-curtis quadrature: {}'.format(len(nodes_clenshaw[1])))


fig1, ax1 = plt.subplots()
ax1.scatter(*nodes_gaussian, marker='o', color='b')
ax1.scatter(*nodes_clenshaw, marker= 'x', color='r')
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')
```

In the following all quadrature rules implemented in chaospy are highlighted:
<table border="1">
<thead>
<tr><th align="center"> Collection of quadrature
rules</th> <th align="center">   Name  </th> <th align="center">Key</th> </tr>
</thead>
<tbody>
<tr><td align="center">   Optimal Gaussian quadrature
</td> <td align="center">   Gaussian     </td> <td align="center">   G
</td> </tr>
<tr><td align="center">   Gauss-Legendre quadrature          </td>
<td align="center">   Legendre     </td> <td align="center">   E      </td>
</tr>
<tr><td align="center">   Clenshaw-Curtis quadrature         </td> <td
align="center">   Clenshaw     </td> <td align="center">   C      </td> </tr>
<tr><td align="center">   Leja quadrature                    </td> <td
align="center">   Leja         </td> <td align="center">   J      </td> </tr>
<tr><td align="center">   Hermite Genz-Keizter 16 rule       </td> <td
align="center">   Genz         </td> <td align="center">   Z      </td> </tr>
<tr><td align="center">   Gauss-Patterson quadrature rule    </td> <td
align="center">   Patterson    </td> <td align="center">   P      </td> </tr>
</tbody>
</table>
It is also possible to use sparse grid quadrature. For this
purpose Clenshaw-Curtis method is advised since it is nested.

In the following
example we show sparse vs. normal quadrature nodes:

```{.python .input}
# example sparse grid quadrature
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

order = 2
# sparse grid has exponential growth, thus a smaller order results in more points
nodes_clenshaw, weights_clenshaw = cp.generate_quadrature(order=order, domain=joint_distribution, rule='C')
nodes_clenshaw_sparse, weights_clenshaw_sparse = cp.generate_quadrature(order=order, domain=joint_distribution, rule='C', sparse=True)

print('Number of nodes normal clenshaw-curtis quadrature: {}'.format(len(nodes_clenshaw[0])))
print('Number of nodes clenshaw-curtis quadrature with sparse grid : {}'.format(len(nodes_clenshaw_sparse[0])))

fig1, ax1 = plt.subplots()
ax1.scatter(*nodes_clenshaw, marker= 'x', color='r')
ax1.scatter(*nodes_clenshaw_sparse, marker= 'o', color='b')
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')
```

### Step 4.c: Polynomial Chaos Expansion

After the model is evaluated for all
integration nodes, the polynomial chaos expansion can be generated with the
following method:

```{.python .input}
# spectral projection in chaospy
cp.fit_quadrature?
```

In the following snippet we show again a complete example for polynomial chaos expansion
using the pseudo spectral approach to calculate the
expansion coefficients.
The
model applied the same simple mathematical expression as before:

<!-- Equation labels as ordinary links -->
<div
id="eq:dummy_model_repeat"></div>

$$
\begin{equation}
 y(z_1, z_2) = z_1 + z_1
z_2
\label{eq:dummy_model_repeat} \tag{11}
\end{equation}
$$

The random variables for $Z_1, Z_2$ are defined as simple uniform random
variables:

<!-- Equation labels as ordinary links -->
<div id="eq:dummy_rv_repeat"></div>
$$
\begin{equation}
Z_1 = \mbox{U}(0,1), \quad Z_2 = \mbox{U}(0,1)
\label{eq:dummy_rv_repeat} \tag{12}
\end{equation}
$$

```{.python .input}
# example spectral projection
# 1. define marginal and joint distributions
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

# 2. generate orthogonal polynomials
polynomial_order = 3
poly = cp.orth_ttr(polynomial_order, joint_distribution)

# 4.1 generate quadrature nodes and weights
order = 5
nodes, weights = cp.generate_quadrature(order=order, domain=joint_distribution, rule='G')

# 4.2 evaluate the simple model for all nodes
model_evaluations = nodes[0]+nodes[1]*nodes[0]

# 4.3 use quadrature to generate the polynomial chaos expansion
gpce_quadrature = cp.fit_quadrature(poly, nodes, weights, model_evaluations)
print("Success")
```

## Step 5: Statistical Analysis
<div id="sec:statistical_analysis"></div>

Once the polynomial chaos expansion has been generated either with **pseudo-spectral projection** or with **regression** method,
the calculation of the statistical properties is straightforward.
The following listing gives an overview of all available
methods take all the same input parameter the **polynomial-expansion** and
the
**joint-distribution** (see also example below).

Note, that one can also
calculate uncertainty statistics on distributions only as well.

### Uncertainty quantification

* Expected value: `cp.E`

* Variance: `cp.Var`

* Standard deviation: `cp.Std`

* Curtosis: `cp.Kurt`

* Skewness: `cp.Skew`

* Distribution of Y: `cp.QoI_Dist`

* Prediction intervals: `cp.Perc`, which is a method to calculate percentiles: an additional argument defining the percentiles
needs to be passed.

If multiple quantities of interest are available:

* Covariance matrix: `cp.Cov`

* Correlation matrix: `cp.Corr`

* Spearman correlation: `cp.Spearman`

* Auto-correlation function: `cp.Acf`

```{.python .input}
# example uq
exp_reg = cp.E(gpce_regression, joint_distribution)
exp_ps =  cp.E(gpce_quadrature, joint_distribution)

std_reg = cp.Std(gpce_regression, joint_distribution)
str_ps = cp.Std(gpce_quadrature, joint_distribution)

prediction_interval_reg = cp.Perc(gpce_regression, [5, 95], joint_distribution)
prediction_interval_ps = cp.Perc(gpce_quadrature, [5, 95], joint_distribution)

print("Expected values   Standard deviation            90 % Prediction intervals\n")
print(' E_reg |  E_ps     std_reg |  std_ps                pred_reg |  pred_ps')
print('  {} | {}       {:>6.3f} | {:>6.3f}       {} | {}'.format(exp_reg,
                                                                  exp_ps,
                                                                  std_reg,
                                                                  str_ps,
                                                                  ["{:.3f}".format(p) for p in prediction_interval_reg],
                                                                  ["{:.3f}".format(p) for p in prediction_interval_ps]))
```

### Sensitivity analysis

The variance bases sensitivity indices can be
calculated directly from the expansion.
The `chaospy` package provides the
following methods:

* first order indices: `cp.Sens_m`

* second order indices: `cp.Sens_m2`

* total indices: `cp.Sens_t`

Here is an example for the first and
total indices for both expansions:

```{.python .input}
# example sens
sensFirst_reg = cp.Sens_m(gpce_regression, joint_distribution)
sensFirst_ps = cp.Sens_m(gpce_quadrature, joint_distribution)

sensT_reg = cp.Sens_t(gpce_regression, joint_distribution)
sensT_ps = cp.Sens_t(gpce_quadrature, joint_distribution)

print("First Order Indices           Total Sensitivity Indices\n")
print('       S_reg |  S_ps                 ST_reg |  ST_ps  \n')
for k, (s_reg, s_ps, st_reg, st_ps) in enumerate(zip(sensFirst_reg, sensFirst_ps, sensT_reg, sensT_ps)):
    print('S_{} : {:>6.3f} | {:>6.3f}         ST_{} : {:>6.3f} | {:>6.3f}'.format(k, s_reg, s_ps, k, st_reg, st_ps))
```
