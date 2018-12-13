# A comparison of Monte Carlo and polynomial chaos approaches for the
sensitivity analysis of the $A1$, $B1$, $B2$ and $C2$ test functions

<!--
dom:AUTHOR: Samuele Lo Piano -->
<!-- Author: -->  
**Samuele Lo Piano**,
[s.lopiano@gmail.com](mailto:s.lopiano@gmail.com)

# The test functions
<div
id="sec:G_functions"></div>

In two previous notebooks
[g_function](interactive_g_function.ipynb) and
[gstar_function](interactive_gstar_function.ipynb), the potential of the
polynomial chaos expansions (with the `chaospy` package) for sensitivity
analysis has been disclosed. Its performance has been benchmarked against the
Monte Carlo methods may be used to approximate the variance based Sobol
sensitivity indices. <br/>

In this notebook, the analysis is extended to the
other [Kucherenko's test
functions](https://www.sciencedirect.com/science/article/pii/S0951832010002437),
for which the usefulness of low-discrepancy sequences towards reaching a higher
convergence pace has been already showcased in [this
notebook](testfunction3.ipynb). We adopt herein the same nomenclature in the
interest of coherency. Functions A2, B3 and C1 are excluded from this notebook
as they have already been implicitly covered in the one dedicated to the
[g_function](interactive_g_function.ipynb). One can get these different cases by
simply tuning the additive $a_i$ coefficients.

<!--
Equation labels as ordinary
links -->
<div id="eq:1"></div>

$$
\begin{equation}
A1: f(X) = \sum_{j=1}^{k}
(-1)^j \, \prod_{l=1}^{j} x_l \label{eq:1} \tag{1}
\end{equation}
$$

<!--
Equation labels as ordinary links
-->
<div id="eq:2"></div>

$$
\begin{equation}
B1: f(X) = \prod_{j=1}^{k} \frac{k-{4x_j}}{k-0.5} \label{eq:2} \tag{2}
\end{equation}
$$

<!-- Equation labels as ordinary links
-->
<div
id="eq:3"></div>

$$
\begin{equation}
B2: f(X) = \left(1+\frac{1}{k}\right)^k
\prod_{j=1}^{k} \sqrt{x_j} \label{eq:3} \tag{3}
\end{equation}
$$

<!--
Equation
labels as ordinary links -->
<div id="eq:4"></div>

$$
\begin{equation}
C2: f(X)
= 2^k
\prod_{j=1}^{k} x_j \label{eq:4} \tag{4}
\end{equation}
$$

All the input
factors $X_i$ are assumed to be uniformly distributed in the interval $[0,1]$.
The number of factors *k* here amounts to six, but it can be varied as the
reader pleases, although it should be never lower than three to produce a
meaningful sensitivity analysis.

Run the first cell to initialise plotting and
printing modules for later use (and system settings).

```{.python .input  n=1}
# ipython magic
%matplotlib notebook
%load_ext autoreload
%autoreload 2
import os, sys, inspect
# Use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"python_source")))
if cmd_subfolder not in sys.path:
     sys.path.insert(0, cmd_subfolder)

%run python_source/matplotlib_header

from present_output import print_vectors_relerror, print_3vectors_relerror
```

As discussed in [the dedicated notebook on test
functions](testfunctions3.ipynb), for functions of type _A_ the sensitivity
indices of the parameters decrease across the series: every parameter is less
influential on the output variance than the previous. Functions of type _A_ have
been precisely designed to represent an asymmetric case of unequal importance
across parameters. Conversely, _B_ and _C_ functions are symmetric with all
parameters equally important. The former do no have significant higher-order
interactions across parameters, having therefore $S_j \approx S_{T_j}$. I.e.,
most of the variance can be explained with the individual parameters first-order
effects. This is not the case for _C_ type functions, for which higher-order
interactions play a prominent role and the inequality $S_{T_j} > S_j$ holds.
The
functions introduced in equations ([1](#eq:1)),  ([2](#eq:2)), ([3](#eq:3))
and
([4](#eq:4)) can be loaded from an external module. Non-standard modules
such as
`sobol_seq` and `chaospy` are required. However, they can be easily
installed by
typing 
        `pip install sobol_seq` 
from your terminal (Linux)
or from the
anaconda-prompt environment in case you use Anaconda package
manager.

```{.python .input}
# import modules
import numpy as np
import pandas as pd
import sobol_seq
from string import ascii_lowercase
from testfunctions import create_dict, A1, B1, B2, C2, AE_dic, AEF_dic
```

The analytical expression of the sensitivity indices can be analytically
computed for all the functions and can be adopted for benchmarking the
performance of the sensitivity indices computed. <br/>
As regards function
$A_1$, 

<!--
Equation labels as ordinary links -->
<div id="eq:5"></div>

$$
\begin{equation}
E_{k2} =
\frac{1}{6}\left(1-\left(\frac{1}{6}\right)^k\right)+\frac{4}{15}\left((-1)^{k+1}\left(\frac{1}{2}\right)^k+\left(\frac{1}{3}\right)^k\right)
\label{eq:5} \tag{5}
\end{equation}
$$

<!--
Equation labels as ordinary links
-->
<div id="eq:6"></div>

$$
\begin{equation}
V(k) =
\frac{1}{10}\left(\frac{1}{3}\right)^k+\frac{1}{18}-\frac{1}{9}\left(\frac{1}{2}\right)^{2k}+(-1)^{k+1}\left(\frac{2}{45}\right)\left(\frac{1}{2}\right)^k
\label{eq:6} \tag{6}
\end{equation}
$$

<!--
Equation labels as ordinary links
-->
<div id="eq:7"></div>

$$
\begin{equation}
E_j(j) =
\frac{1}{6}\left(1-\left(\frac{1}{3}\right)^j\right)+\frac{4}{15}\left((-1)^{j+1}\left(\frac{1}{2}\right)^j+\left(\frac{1}{3}\right)^j\right)
\
\label{eq:7} \tag{7}
\end{equation}
$$

<!--
Equation labels as ordinary links
-->
<div id="eq:8"></div>

$$
\begin{equation}
T_1 =
\frac{1}{2}\left(\frac{1}{3}^{j-1}\right)\left(1-\left(\frac{1}{3}\right)^{k-j}\right)
\
\label{eq:8} \tag{8}
\end{equation}
$$

<!--
Equation labels as ordinary links
-->
<div id="eq:9"></div>

$$
\begin{equation}
T_2 =
\frac{1}{2}\left(\frac{1}{3}^j-\frac{1}{3}^k\right) \
\label{eq:9} \tag{9}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div
id="eq:10"></div>

$$
\begin{equation}
T_3 =
\frac{3}{5}\left(4\left(\frac{1}{3}\right)^{k+1}+(-1)^{j+k+1}\left(\frac{1}{2}\right)^{k-j+2}+\left(\frac{1}{3}\right)^{j+1}\right)
\
\label{eq:10} \tag{10}
\end{equation}
$$

<!--
Equation labels as ordinary
links -->
<div id="eq:11"></div>

$$
\begin{equation}
T_4 =
\frac{1}{5}\left((-1)^{j+2}\frac{1}{3}\left(\frac{1}{2}\right)^{j-2}-4\left(\frac{1}{3}\right)^{j+1}\right)
\
\label{eq:11} \tag{11}
\end{equation}
$$

<!--
Equation labels as ordinary
links -->
<div id="eq:12"></div>

$$
\begin{equation}
T_5 =
\frac{1}{5}\left((-1)^{k+1}\frac{1}{3}\left(\frac{1}{2}\right)^{k-2}+(-1)^{k+j}\left(\frac{1}{3}\right)^{j+1}\left(\frac{1}{2}\right)^{k-j-2}\right)
\
\label{eq:12} \tag{12}
\end{equation}
$$

From which one can derive the
expression for the total-order sensitivity indices $S_{T_j}$ as:

<!--
Equation
labels as ordinary links -->
<div id="eq:13"></div>

$$
\begin{equation}
S_{T_j}
= \frac{E_{k2}(k)-E_j(j)-\frac{1}{4}(T_1(j)-2T_2(j)+T_3(j)-T_4(j)-T_5(j))}{V(k)}
\
\label{eq:13} \tag{13}
\end{equation}
$$

As well as first order indices:
<!--
Equation labels as ordinary links -->
<div id="eq:14"></div>

$$
\begin{equation}
S_j =
\frac{E_{k2}(k)-E_j(j)-\frac{1}{4}(T_1(j)-2T_2(j)+T_3(j)-T_4(j)-T_5(j))\left(12\left(1-\left(\frac{1}{2}\right)^k\right)^2\right)}{V(k)*27\left(\frac{1}{2}-\frac{4}{5}\left(\frac{-1}{2}\right)^k+\frac{3}{10}\left(\frac{1}{3}\right)^k\right)}
\
\label{eq:14} \tag{14}
\end{equation}
$$

The analytical expressions for the
other functions are luckily less complicated. For function $B_1$:

<!--
Equation
labels as ordinary links -->
<div id="eq:15"></div>

$$
\begin{equation}
p(j) =
12(k-0.5)^2
\label{eq:15} \tag{15}
\end{equation}
$$

<!--
Equation labels as
ordinary links -->
<div id="eq:16"></div>

$$
\begin{equation}
S_{T_j} =
\frac{(p(j)+1)^k}{(p(j)+1)((p(j)+1)^k-p(j)^k)}
\label{eq:16} \tag{16}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div
id="eq:17"></div>

$$
\begin{equation}
S_j =
\frac{1}{(p(j)\left(1+\left(\frac{1}{p(j)}\right)^k\right)}
\label{eq:17}
\tag{17}
\end{equation}
$$

And for function $B_2$:

<!--
Equation labels as
ordinary links -->
<div id="eq:18"></div>

$$
\begin{equation}
S_{T_j} =
\frac{(k+1)^(2k-2)}{(k+1)^2k-(k^2+2k)^k}
\label{eq:18} \tag{18}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div id="eq:19"></div>

$$
\begin{equation}
S_j =
\frac{1}{(k^2+2k)\left(1+\left(\frac{1}{k^2+2k}\right)\right)^{k-1}}
\label{eq:19} \tag{19}
\end{equation}
$$

Finally, for function $C_2$:

<!--
Equation labels as ordinary links -->
<div id="eq:20"></div>

$$
\begin{equation}
S_{T_j} = \frac{4^{k-1}}{4^k-3^k}
\label{eq:20} \tag{20}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div
id="eq:21"></div>

## Comparison of the sensitivity indices computed with the
spectral expansions and the Monte Carlo simulations

In this section, we
benchmark the performance of the two approaches against the analytical values
dependent on the sample size explored. One can firstly select one of the four
test functions. This comparison has been undertaken for the
[g_function](interactive_g_function.ipynb) and
[gstar_function](interactive_gstar_function.ipynb) in dedicated notebooks.

```{.python .input}
def f(x):
    return x

import ipywidgets as widgets
chart = widgets.interactive(f, x=['A1','B1','B2','C2'])
display(chart)

k = 6
```

The number of parameters _k_ has been here set as three, but the interested
reader can vary it as s/he pleases. However, one should bear in mind a
meaningful sensitivity analysis needs at least three parameters.

```{.python .input}
f = eval(chart.children[0].value)
```

```{.python .input}
Siv = []
for ke in AEF_dic.keys(): #update the functions analytical values
    if ke[2:4] == chart.children[0].value:
        Siv.append(AEF_dic[ke])
Si = np.array(Siv)

STv = []
for kef in AE_dic.keys():
    if kef[2:4] == chart.children[0].value:
        STv.append(AE_dic[kef])
ST = np.array(STv)
```

```{.python .input}
# chaospy f-function with sliders
import chaospy as cp

if not 'jpdf' in globals():
    jpdf = cp.Iid(cp.Uniform(),k) #the joint pdf
    print('Create the joint pdf')


def update_f(**kwargs):
    NsPC=kwargs['NsPC']
    del kwargs['NsPC']
    NsMC=kwargs['NsMC']
    del kwargs['NsMC']
    
    polynomial_order=kwargs['polynomial_order']
    del kwargs['polynomial_order']

    prm_cat=int(len(kwargs)/k)
    prms=np.zeros((prm_cat,k))

    import re
    r = re.compile("([a-zA-Z]+)([0-9]+)")
 
    for key, value in kwargs.items(): #find indx and value for a_prms
        pre,post=r.match(key).groups()
        cat_idx=strings.index(pre)
        prms[cat_idx,int(post)]=value
    
    ## Monte Carlo update
    print('Number of samples for Monte Carlo: ', NsMC) 
    X_mc=sobol_seq.i4_sobol_generate(2*k,NsMC)
    A = X_mc[:,:k] # Generate A and B matrices
    B = X_mc[:,k:]
    f_A_sample = f(pd.DataFrame(A)).values
    f_B_sample = f(pd.DataFrame(B)).values
    
    exp_mc = np.mean(f_A_sample)
    std_mc = np.std(f_A_sample,ddof=0)
    print("Statistics Monte Carlo\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('mc  : {:2.5f} | {:2.5f}'.format(float(exp_mc), std_mc))
    
    A_mixed = []
    for j in range(k):
        A_mixed.append(A.copy())
        A_mixed[j][:,j] = B[:,j]
    
    f_A_mixed = np.array([f(pd.DataFrame(As)) for As in A_mixed])
    S_mc = np.array([np.mean(f_B_sample*(f_A_mixed[j]-f_A_sample))/np.var(f_A_sample,ddof=0) for j in range(k)])
    S_tmc = np.array([0.5*np.mean((f_A_mixed[j]-f_A_sample)**2)/np.var(f_A_sample,ddof=0) for j in range(k)])

    Xpc=sobol_seq.i4_sobol_generate(k,NsPC)
    print('Number of samples: ',NsPC)

    f_sample=f(pd.DataFrame(Xpc)).values

    poly = cp.orth_ttr(polynomial_order, jpdf)
    approx = cp.fit_regression(poly, Xpc.T, f_sample)

    exp_pc = cp.E(approx, jpdf)
    std_pc = cp.Std(approx, jpdf)
    print("Statistics polynomial chaos\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('pc  : {:2.5f} | {:2.5f}'.format(float(exp_pc), std_pc))
    S_pc = cp.Sens_m(approx, cp.Iid(cp.Uniform(),k)) #Si from chaospy
    S_tpc = cp.Sens_t(approx, cp.Iid(cp.Uniform(),k)) #Total effect sensitivity index from chaospy
    
    row_labels= ['S_'+str(idx) for idx in range(k)]
    #col_labels=['Chaospy','Analytical','Error (%)']
    col_labels=['Monte Carlo','Err (%)','PolyChaos','Err (%)']

    print("\nFirst Order Indices")
#    print_vectors_relerror(S_pc,Si,col_labels,row_labels,[3,3,0])
    print_3vectors_relerror(S_mc,S_pc, Si, col_labels, row_labels, [3,0,3,0])

    print("\n\nTotal Effect Indices")
    row_labels= ['St_'+str(idx) for idx in range(k)]
#    print_vectors_relerror(S_tpc,ST,col_labels,row_labels,[3,3,0])
    print_3vectors_relerror(S_tmc,S_tpc, ST, col_labels, row_labels, [3,0,3,0])


my_sliders=[]

my_sliders.append(widgets.IntSlider(min=500,max=5100,step=250,value=500,description='NsPC')) #add slider for samples
my_sliders.append(widgets.IntSlider(min=500,max=5100,step=250,value=500,description='NsMC')) #add slider for samples
my_sliders.append(widgets.IntSlider(description='polynomial_order', min=1,max=6,value=4)) # add slider for polynomial order

slider_dict = {slider.description:slider for slider in my_sliders} #add the sliders in the dictionary 

ui_left = widgets.VBox(my_sliders[0::3]) 
ui_mid  = widgets.VBox(my_sliders[1::3])
ui_right = widgets.VBox(my_sliders[2::3])
ui=widgets.HBox([ui_left,ui_mid,ui_right])

out=widgets.interactive_output(update_f, slider_dict) 
display(ui,out)

# end chaospy G-function with sliders
```

What approach converges the quickest towards the analytical value? What indices
are the hardest to compute and why is this the case? Do you think this
occurrence may negatively impact on the quality of the analysis in a real case
study?
