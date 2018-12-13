# Global Sensitivity Analysis with test functions

<!-- AUTHOR: Samuele lo Piano
-->
**Samuele Lo Piano**, [s.lopiano@gmail.com](mailto:s.lopiano@gmail.com)

```{.python .input}
# ipython magic
%matplotlib notebook
%load_ext autoreload
%autoreload 2
```

```{.python .input}
%matplotlib inline

# plot configuration
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# import seaborn as sns # sets another style
matplotlib.rcParams['lines.linewidth'] = 3
fig_width, fig_height = (7.0,5.0)
matplotlib.rcParams['figure.figsize'] = (fig_width, fig_height)

# font = {'family' : 'sans-serif',
#   'weight' : 'normal',
#   'size' : 18.0}
# matplotlib.rc('font', **font) # pass in the font dict as kwar
```

[Global sensitivity
analysis](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184) is an
approached aimed at correlating the uncertainty in the output to the uncertainty
in the input parameters. When dealing with non-additive models, it outperforms
one-variable-at-a-time sensitivity analysis as this latter does not account for
high-order interactions among parameters. This results in incorrect apportion of
output uncertainty onto input parameters.

A statistical property typically used
to account for uncertainty is variance, leading to variance-based sensitivity
analysis as a largely adopted practice.

It may usually happen in models that
not all the input-parameters variability equally affects the output variability.
Typical trends consist in [Pareto
distributions](https://en.wikipedia.org/wiki/Pareto_distribution) having a small
share of the input variables responsible for most of the output variance. Yet
one may come across other situations with important high-order interactions
cutting across (almost) all the input parameters.

The seven analytical
functions studied in
[Kucherenko_et_al_2011](https://www.sciencedirect.com/science/article/pii/S0951832010002437/)
have precisely this aim: represent the possible cases one may come across when
computing sensitivity indices. Three typical situations are present: A-type
functions with not equally important variables, where one can identify a small
set of leading variables while the remainder playing the role of complementary
variables; B-type functions, having (normally more than one) dominant low-order
terms; C-type functions, for which (almost) all of the parameters are important
with significant high-order interactions.

Plenty of algorithms have been
proposed to compute estimators that could adequately account for the above-
mentioned sensitivity indices. Exact analytical expressions are available for
these analytical functions, hance putting one in the condition to estimate the
quality of the assessment performed with these estimators. One can measure the
convergence rate and see how quickly an estimator approaches the actual
sensitivity indices value by enlarging the sample size.

As we shall see in this
notebook, the convergence rate is lower for C-type functions over B-type
functions and finally A-type function, these latter presenting the quickest
convergence. We also compare here the convergence pace of samples based on [Low-
discrepancy Sobol sequency](sobol_interactive.ipynb) against standard random
numbers. More details in the [dedicated appendix](sobol_interactive.ipynb).

## Defining the test functions

<div id="sec:the_test_functions"></div>
One can find the expression of the seven test functions we will be dealing within this notebook in equation 1 ("eq:equation 1": "(#eq:test_functions)" below

<!--
Equation labels as ordinary links -->
<div id="_auto1"></div>

$$
\begin{equation}
A1: f(X) = \sum_{j=1}^{k} (-1)^j \, \prod_{l=1}^{j} x_l \
\label{_auto1} \tag{1}
\end{equation}
$$

<!-- Equation labels as ordinary links
-->
<div id="_auto2"></div>

$$
\begin{equation}
A2: f(X) = \prod_{j=1}^{k}
\frac{|{4x_j}-2|+{a_j}}{1+{a_j}} \
\label{_auto2} \tag{2}
\end{equation}
$$
<!-- Equation labels as ordinary links -->
<div id="_auto3"></div>

$$
\begin{equation}
B1: f(X) = \prod_{j=1}^{k} \frac{k-{4x_j}}{k-0.5} \
\label{_auto3} \tag{3}
\end{equation}
$$

<!-- Equation labels as ordinary links
-->
<div id="_auto4"></div>

$$
\begin{equation}
B2: f(X) = \left(1+\frac{1}{k}\right)^k
\prod_{j=1}^{k} \sqrt{x_j} \
\label{_auto4} \tag{4}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div id="_auto5"></div>

$$
\begin{equation}
B3: f(X) = \prod_{j=1}^{k} \frac{|{4x_j}-2|+{b_j}}{1+{b_j}} \
\label{_auto5} \tag{5}
\end{equation}
$$

<!-- Equation labels as ordinary links
-->
<div id="_auto6"></div>

$$
\begin{equation}
C1: f(X) = \prod_{j=1}^{k}
|{4x_j}-2|\
\label{_auto6} \tag{6}
\end{equation}
$$

<!-- Equation labels as
ordinary links -->
<div id="_auto7"></div>

$$
\begin{equation}
C2: f(X) = 2^k
\prod_{j=1}^{k} x_j \
\label{_auto7} \tag{7}
\end{equation}
$$

In our case we
will be working with a set of six parameters. The reader may change the pool as
s/he pleases along with the dimension of the lists of the *a* and *b*
coefficients. The minimal parameter-pool size able to produce a meaningful
inference is set at three.

```{.python .input}
k = 6 # k represents the number of parameters inquired

# a and b are the constants' vectors used in the G-function expressions
a = [0, 0, 6.52, 6.52, 6.52, 6.52]
b = [6.52, 6.52, 6.52, 6.52, 6.52, 6.52]
```

The *a* list allows to cover the case where one has two leading parameters with
the remaining four in the position of complementarity. Conversely, the *b* list
represents the case where all the parameters are equally important. In general,
the lower the constant !bt a_k !et, the higher the importance of a parameter for
the *G functions* A2, B3 and C1 presented in the study. C1 is the most
challenging one as all the additive constants are set at 0.

The analytical
formulae for the values of the sensitivity indices were retrieved from the
quoted
[kucherenko_2011](https://www.sciencedirect.com/science/article/pii/S0951832010002437/)
or
[saltelli_annoni_2010](https://www.sciencedirect.com/science/article/pii/S0010465509003087/)
(for the functions A1, A2 and B3). The functions analytical values will serve as
external referent when testing the estimators convergence pace.

A dataframe of
low-discrepancy quasi-random numbers in the interval [0,1] of convenient size is
then generated by using a specific algorithm, the [Sobol
sequence](https://en.wikipedia.org/wiki/Sobol_sequence). The sample size is
defined in relation to powers of 2 due to the specific properties of this
algorithm.

The import of a non-standard module, 'sobol_seq', is required. It
can be easily installed by typing `pip install sobol_seq` from your terminal
(Linux) or from the anaconda-prompt environment in case you use Anaconda package
manager.

```{.python .input}
# import modules
import numpy as np
import pandas as pd
import sobol_seq
from string import ascii_lowercase
from testfunctions import create_dict, functions, AE_dic, AEF_dic
```

```{.python .input}
letters = [l for l in ascii_lowercase]

p_sample = []
p_sampleR = []
p_sample_name = []
f_sample = []
f_sampleR = []
p = 12

df = pd.DataFrame(sobol_seq.i4_sobol_generate(2*k, 2**p-4))
df2 = pd.DataFrame(np.random.rand(2**p-4,k*2))
```

```{.python .input}
qamples = []
qamplesr = []
for s in range (2,p):
    qamples.append(df.iloc[(-4+2**s):(-4+2**(s+1))].reset_index(drop=True))
    qamplesr.append(df2.iloc[(-4+2**s):(-4+2**(s+1))].reset_index(drop=True))

    CheckMAE_mean = []
    CheckMAER_mean = []
    CheckMAEF_mean = []
    CheckMAEFR_mean = []

    sample_Matrices = []
    sample_MatricesR = []
    for m in range (0, 2):
        sample_Matrices.append(qamples[s-2].T.iloc[int(m*(len(qamples[s-2].columns)/2)):int((m+1)*(len(qamples[s-2].columns)/2))].reset_index(drop=True).T)
        sample_MatricesR.append(qamplesr[s-2].T.iloc[int(m*(len(qamplesr[s-2].columns)/2)):int((m+1)*(len(qamplesr[s-2].columns)/2))].reset_index(drop=True).T)

    sample_Matrices_dic = create_dict(letters, sample_Matrices)
    sample_MatricesR_dic = create_dict(letters, sample_MatricesR)

    mixed_Matrices = []
    mm_names = []
    mixed_MatricesR = []
    for sm in range (0,len(sample_Matrices)):
        for sm1 in range (0,len(sample_Matrices)):
            if sm == sm1:
                continue
            else:
                for c in sample_Matrices[sm]:
                    mixed_Matrices.append(sample_Matrices[sm].copy())
                    mixed_Matrices[len(mixed_Matrices)-1][c]=sample_Matrices[sm1].copy()[c]
                    mm_names.append(str(letters[sm] + letters[sm1] + str(c+1)))
                    mixed_MatricesR.append(sample_MatricesR[sm].copy())
                    mixed_MatricesR[len(mixed_MatricesR)-1][c]=sample_MatricesR[sm1].copy()[c]

    mixed_Matrices_dic = create_dict(mm_names, mixed_Matrices)
    mixed_MatricesR_dic = create_dict(mm_names, mixed_MatricesR)

    matrices_dic = {**sample_Matrices_dic, **mixed_Matrices_dic}
    matricesR_dic = {**sample_MatricesR_dic, **mixed_MatricesR_dic}

    names1 = []
    values1R = []
    values1 = []
    names2 = []
    values2 = []
    values2R = []
    for f in functions:
        for sq, zq in mixed_Matrices_dic.items():
            names1.append(f.__name__+str(sq))
            values1.append(f(zq))
        for sqR, zqR in mixed_MatricesR_dic.items():
            values1R.append(f(zqR))

        for sM, zM in matrices_dic.items():
            names2.append(f.__name__+str(sM))
            values2.append(f(zM))
        for sMR, zMR in matricesR_dic.items():
            values2R.append(f(zMR))

    f_MM_dic = create_dict(names1, values1)
    f_matrices_dic = create_dict(names2, values2)
    f_MMR_dic = create_dict(names1, values1R)
    f_matricesR_dic = create_dict(names2, values2R)

    Check=[]
    CheckR=[]
    CheckName = []
    Check3=[]
    Check3R=[]
    Check3Name = []
    for f in functions:
        for j in range(1,k+1):
            difference = []
            difference3 = []
            differenceR = []
            difference3R = []
            for mk, mz in f_matrices_dic.items():
                if mk[0:2]==f.__name__:
                    validkeys = []
                    validkeys3 = []
                    for fk1 in f_MM_dic.keys():
                        if len(mk)==3 and mk[2]=='a': 
                            if fk1[0:3]==mk[0:3] and fk1[-1]==str(j):
                                validkeys.append(fk1)
                            if fk1[0:2]==mk[0:2] and fk1[2]!=mk[2] and fk1[-1]==str(j):
                                validkeys3.append(fk1)
                    z1 = dict(filter(lambda i:i[0] in validkeys, f_MM_dic.items()))
                    z3 = dict(filter(lambda i3:i3[0] in validkeys3, f_MM_dic.items()))
                    for zk, zv in z1.items():
                        difference.append(0.5*(((mz-zv)**2).mean())/mz.var())
                    for zk3, zv3 in z3.items():
                        difference3.append(((mz*zv3).mean()-mz.mean()**2)/mz.var())
            Check.append(sum(difference)/len(difference))
            CheckName.append('Jansen'+ str(f.__name__) +'ST'+str(j))
            Check3.append(sum(difference3)/len(difference3))
            Check3Name.append('Sobol'+ str(f.__name__) +'S'+str(j))
            for mkR, mzR in f_matricesR_dic.items():
                if mkR[0:2]==f.__name__:
                    validkeysR = []
                    validkeys3R = []
                    for fk1R in f_MMR_dic.keys():
                        if len(mkR)==3 and mkR[2]=='a': 
                            if fk1R[0:3]==mkR[0:3] and fk1R[-1]==str(j):
                                validkeysR.append(fk1R)
                            if fk1R[0:2]==mkR[0:2] and fk1R[2]!=mkR[2] and fk1R[-1]==str(j):
                                validkeys3R.append(fk1R)
                    z1R = dict(filter(lambda iR:iR[0] in validkeysR, f_MMR_dic.items()))
                    z3R = dict(filter(lambda iR3:iR3[0] in validkeys3R, f_MMR_dic.items()))
                    for zkR, zvR in z1R.items():
                        differenceR.append(0.5*(((mzR-zvR)**2).mean())/mzR.var())
                    for zk3R, zv3R in z3R.items():
                        difference3R.append(((mzR*zv3R).mean()-mzR.mean()**2)/mzR.var()) 
            CheckR.append(sum(differenceR)/len(differenceR))
            Check3R.append(sum(difference3R)/len(difference3R))
    Check_dic = create_dict(CheckName, Check)
    Check3_dic = create_dict(Check3Name, Check3)
    CheckR_dic = create_dict(CheckName, CheckR)
    Check3R_dic = create_dict(Check3Name, Check3R)

    CheckMAEs = []
    CheckMAEsR = []
    CheckMAENames = []
    CheckMAEsF = []
    CheckMAEsFR = []
    CheckMAEFNames = []
    for ae, av in AE_dic.items():
        for Lk, Lv in Check_dic.items():
            if ae[-5:]==Lk[-5:]:
                CheckMAEs.append(abs(Lv-av))
                CheckMAENames.append('CheckMAE'+ str(ae[2:4]) + str(ae[-1]))
        for LkR, LvR in CheckR_dic.items():
            if ae[-5:]==LkR[-5:]:
                CheckMAEsR.append(abs(LvR-av))
    for af, afv in AEF_dic.items():
        for Lk3, Lv3 in Check3_dic.items():
            if af[-4:]==Lk3[-4:]:
                CheckMAEsF.append(abs(Lv3-afv))
                CheckMAEFNames.append('CheckMAE'+ str(af[2:4]) + str(af[-1]))
        for Lk3R, Lv3R in Check3R_dic.items():
            if af[-4:]==Lk3R[-4:]:
                CheckMAEsFR.append(abs(Lv3R-afv))
    CheckMAEs_dic = create_dict(CheckMAENames, CheckMAEs)
    CheckMAEsF_dic = create_dict(CheckMAEFNames, CheckMAEsF)
    CheckMAEsR_dic = create_dict(CheckMAENames, CheckMAEsR)
    CheckMAEsFR_dic = create_dict(CheckMAEFNames, CheckMAEsFR)

    CheckMAE = []
    CheckMAER = []
    CheckMAE_name = []
    CheckMAEF = []
    CheckMAEFR = []
    for f in functions:
        validkeys2 = []
        validkeys4 = []
        validkeys2R = []
        validkeys4R = []
        for Lmk, Lmv in CheckMAEs_dic.items():
            if Lmk[-3:-1]==f.__name__:
                 validkeys2.append(Lmk)
        for Fmk, Fmv in CheckMAEsF_dic.items():
            if Fmk[-3:-1]==f.__name__:
                 validkeys4.append(Fmk)
        z2 = dict(filter(lambda i2:i2[0] in validkeys2, CheckMAEs_dic.items()))
        z4 = dict(filter(lambda i4:i4[0] in validkeys4, CheckMAEsF_dic.items()))
        CheckMAE.append(sum(z2.values())/len(z2))
        CheckMAE_name.append('CheckMAE'+f.__name__)
        CheckMAEF.append(sum(z4.values())/len(z4))
        for LmkR, LmvR in CheckMAEsR_dic.items():
            if LmkR[-3:-1]==f.__name__:
                validkeys2R.append(LmkR)
        for FmkR, FmvR in CheckMAEsFR_dic.items():
            if FmkR[-3:-1]==f.__name__:
                 validkeys4R.append(FmkR)
        z2R = dict(filter(lambda i2R:i2R[0] in validkeys2R, CheckMAEsR_dic.items()))
        z4R = dict(filter(lambda i4R:i4R[0] in validkeys4R, CheckMAEsFR_dic.items()))
        CheckMAER.append(sum(z2R.values())/len(z2R))
        CheckMAEFR.append(sum(z4R.values())/len(z4R))
    CheckMAE_dic = create_dict(CheckMAE_name, CheckMAE)
    CheckMAEF_dic = create_dict(CheckMAE_name, CheckMAEF)
    CheckMAER_dic = create_dict(CheckMAE_name, CheckMAER)
    CheckMAEFR_dic = create_dict(CheckMAE_name, CheckMAEFR)
    CheckMAE_mean.append(CheckMAE_dic)
    CheckMAER_mean.append(CheckMAER_dic)
    CheckMAEF_mean.append(CheckMAEF_dic)
    CheckMAEFR_mean.append(CheckMAEFR_dic)
    CheckMAE_mean_dic = {Lmk1:[CheckMAE_mean[Lmv1][Lmk1] for Lmv1 in range(len(CheckMAE_mean))] for Lmk1 in CheckMAE_mean[0].keys()}
    CheckMAER_mean_dic = {Lmk2:[CheckMAER_mean[Lmv2][Lmk2] for Lmv2 in range(len(CheckMAER_mean))] for Lmk2 in CheckMAER_mean[0].keys()}
    CheckMAEF_mean_dic = {Lmk3:[CheckMAEF_mean[Lmv3][Lmk3] for Lmv3 in range(len(CheckMAEF_mean))] for Lmk3 in CheckMAEF_mean[0].keys()}
    CheckMAEFR_mean_dic = {Lmk4:[CheckMAEFR_mean[Lmv4][Lmk4] for Lmv4 in range(len(CheckMAEFR_mean))] for Lmk4 in CheckMAEFR_mean[0].keys()}
    for Lmk1,Lmv1 in CheckMAE_mean_dic.items():
        CheckMAE_mean_dic[Lmk1] = sum(CheckMAE_mean_dic[Lmk1])/len(CheckMAE_mean_dic[Lmk1])
    for Lmk2,Lmv2 in CheckMAER_mean_dic.items():
        CheckMAER_mean_dic[Lmk2] = sum(CheckMAER_mean_dic[Lmk2])/len(CheckMAER_mean_dic[Lmk2])
    for Lmk3,Lmv3 in CheckMAEF_mean_dic.items():
        CheckMAEF_mean_dic[Lmk3] = sum(CheckMAEF_mean_dic[Lmk3])/len(CheckMAEF_mean_dic[Lmk3])
    for Lmk4,Lmv4 in CheckMAEFR_mean_dic.items():
        CheckMAEFR_mean_dic[Lmk4] = sum(CheckMAEFR_mean_dic[Lmk4])/len(CheckMAEFR_mean_dic[Lmk4])

    p_sample.append(CheckMAE_mean_dic)
    p_sample_name.append(2**s)
    f_sample.append(CheckMAER_mean_dic)
    p_sampleR.append(CheckMAEF_mean_dic)
    f_sampleR.append(CheckMAEFR_mean_dic)
    p_sample_dic = create_dict(p_sample_name, p_sample)
    f_sample_dic = create_dict(p_sample_name, f_sample)
    p_sampleR_dic = create_dict(p_sample_name, p_sampleR)
    f_sampleR_dic = create_dict(p_sample_name, f_sampleR)

Checkq = pd.DataFrame.from_dict(p_sample_dic)
Checkf = pd.DataFrame.from_dict(f_sample_dic)
CheckR = pd.DataFrame.from_dict(p_sampleR_dic)
CheckfR = pd.DataFrame.from_dict(f_sampleR_dic)
```

The Sobol matrix gets sliced in a way that every sample is twice bigger than the
previous `qamples` list elements. The first one has only two rows. From
iteration to iteration the number gets increased as 2 to the power of s, up to
2,048 rows for the last one. The `qamples` matrix gets in turn sliced in 50
parts of equivalent size. The rationale behind this operation is generating an
adequate numbers of runs whose output can be averaged to compensate for
fluctuations.

Two sample matrices are generated by slicing down the larger
matrix in two parts of equal length. The code line `sample_Matrices_dic` creates
a dictionary by appending the name to the matrices in the order they have been
generated (a, b). This operation can be ideally replicated in the case one
wishes to compute more sample matrices for the estimators algorithm.

The mixed
matrices (i.e. ab1, ab6, ba2, etc.) are generated by scrambling the columns of
the sample matrices. The first *if* excludes counting the same matrix twice, the
second lower level *for* loop replicates the matrices the number of parameters
one has and for each scrambles the relative column. Finally, the name is
appended dependent the original matrix, the matrix whose column has been used
for the scrambling and the number of the column. The label is eventually
associated to the mixed matrices in the same way as done for the sample
matrices. The sample and mixed matrices are zipped together into a
`matrices_dic`.

The functions are applied to the sets of matrices and the
result of the operations are stored into dictionaries again. Now each item in
the dictionaries `f_MM_dic` and `f_matrices_dic` is a vector rather than a
matrix (f(a), f(b), f(ab4), etc.)

For each function, the sample matrices are
selected ('if len(mk)==3:'). From the ensemble of the scramble matrices, those
having the same function and starting letter are selected for the total
sensivity index accounting. While a different starting letter is the condition
for the first-order estimator (fk1[2]!=mk[2]). The selection is based on the
names (keys). The dictionary is filtered according to these criteria and the
Jansen estimator-to-variance ratio is computed for the higher order (`Check` and
`CheckR`) and the Sobol estimator-to-variance ratio for the first order matrices
(`Check3` and `Check3R`).

Finally, the Mean Absolute Errors, the difference
between the analytical value and the estimator figure, are computed and the
values appended in a dictionary.

These figures are then averaged on the
different parameters (from six figures to one) as well as runs and stored in a
dictionary `CheckMAE_dic` and `CheckMAEF_dic`.

Which is in turn appended onto
the final dictionary wrapping up all the experiments performed (up to the power
of 2 'p' initially defined). This latter is eventually converted into a
convenient *pandas* dataframe from which the trends can be easily plotted or
statistically inference on the figures can be produced.

```{.python .input}
for ind, row in Checkq.iterrows():
 for indR, row in CheckR.iterrows():
  if ind == indR:
   x_vals = Checkq.columns.values
   y1 = Checkq.loc[ind]
   y2 = CheckR.loc[indR]
   for i6 in range(0, len(x_vals), 1):
    plt.loglog(x_vals[i6:i6+2], y1[i6:i6+2], c = "b", marker = "o", label = 'QR' if i6 == 0 else '')
    plt.loglog(x_vals[i6:i6+2], y2[i6:i6+2], c = "r", marker = "x", label ='R' if i6 == 0 else '')
    plt.xlim(0,8500)
   plt.title('Jansen' + '_ST_' +str(ind[-2:]))
   plt.legend()
   plt.show()
```

A series of plots is finally produced whose trend is compared against the sample
size across functions in order to evaluate the convergence rate of the higher-
order sensitivity indices. The convergence paces appear to be slower for C-type
functions. For these latter, the convergence pace of Quasi-random-based
functions against random-based functions appears to be similar and there would
be no advantage in using a low-discrepancy sequence.

```{.python .input}
for ind, row in Checkf.iterrows():
 for indR, row in CheckfR.iterrows():
  if ind == indR:
   x_vals = Checkf.columns.values
   y1 = Checkf.loc[ind]
   y2 = CheckfR.loc[indR]
   for i6 in range(0, len(x_vals), 1):
    plt.loglog(x_vals[i6:i6+2], y1[i6:i6+2], c = "b", marker = "o", label = 'QR' if i6 == 0 else '')
    plt.loglog(x_vals[i6:i6+2], y2[i6:i6+2], c = "r", marker = "x", label ='R' if i6 == 0 else '')
    plt.xlim(0,8500)
   plt.title('Sobol' + '_S_' +str(ind[-2:]))
   plt.legend()
   plt.show()
```

The same operation can be repeated for first-order indices. The inference on the
trends is analogous to high-order indices.

## Conclusions

The convergence pace
is an aspect an analyst should have clear in mind when designing the experiment
to account for the sensitivity indices: complicated models with plenty of
relevant higher-order interactions would call for larger sample sizes and
therefore longer and more computational demanding experiments to adequately
estimate the sensitivity indices. Low-discrepancy sequences can play a role in
easing the correct computation of sensitivity indices as long as high-order
terms are not dominant for most of the input parameters.
