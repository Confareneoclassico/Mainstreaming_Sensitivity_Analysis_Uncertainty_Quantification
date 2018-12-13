# A comparison of Monte Carlo and polynomial chaos approaches for the sensitivity analysis of the $A1$, $B1$, $B2$ and $C2$ test functions

<!-- dom:AUTHOR: Samuele Lo Piano -->
<!-- Author: -->  
**Samuele Lo Piano**, [s.lopiano@gmail.com](mailto:s.lopiano@gmail.com)

# The test functions
<div id="sec:G_functions"></div>

In two previous notebooks [g_function](interactive_g_function.ipynb) and [gstar_function](interactive_gstar_function.ipynb), the potential of the polynomial chaos expansions (with the `chaospy` package) for sensitivity analysis has been disclosed. Its performance has been benchmarked against the Monte Carlo methods may be used to approximate the variance based Sobol sensitivity indices. <br/>

In this notebook, the analysis is extended to the other [Kucherenko's test functions](https://www.sciencedirect.com/science/article/pii/S0951832010002437), for which the usefulness of low-discrepancy sequences towards reaching a higher convergence pace has been already showcased in [this notebook](testfunction3.ipynb). We adopt herein the same nomenclature in the interest of coherency. Functions A2, B3 and C1 are excluded from this notebook as they have already been implicitly covered in the one dedicated to the [g_function](interactive_g_function.ipynb). One can get these different cases by simply tuning the additive $a_i$ coefficients.

<!--
Equation labels as ordinary links -->
<div id="eq:1"></div>

$$
\begin{equation}
A1: f(X) = \sum_{j=1}^{k} (-1)^j \, \prod_{l=1}^{j} x_l \
\label{_auto1} \tag{1}
\end{equation}
$$

<!-- Equation labels as ordinary links
-->
<div id="eq:2"></div>

$$
\begin{equation}
B1: f(X) = \prod_{j=1}^{k} \frac{k-{4x_j}}{k-0.5} \
\label{_auto3} \tag{3}
\end{equation}
$$

<!-- Equation labels as ordinary links
-->
<div id="eq:3"></div>

$$
\begin{equation}
B2: f(X) = (1+\frac{1}{n})^n
\prod_{j=1}^{k} \sqrt{x_j} \
\label{_auto4} \tag{4}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div id="eq:4"></div>

$$
\begin{equation}
C2: f(X) = 2^k
\prod_{j=1}^{k} x_j \
\label{_auto7} \tag{7}
\end{equation}
$$

All the input factors $X_i$ are assumed to be uniformly distributed in the interval $[0,1]$. The number of factors *k* here amounts to six, but it can be varied as the reader pleases, although it should be never lower than three to produce a meaningful sensitivity analysis.

Run the first cell to initialise plotting and printing modules for later use (and system settings).

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

As discussed in [the dedicated notebook on test functions](testfunctions3.ipynb), for functions of type _A_ the sensitivity indices of the parameters decrease across the series: every parameter is less influential on the output variance than the previous. Functions of type _A_ have been precisely designed to represent an asymmetric case of unequal importance across parameters. Conversely, _B_ and _C_ functions are symmetric with all parameters equally important. The former do no have significant higher-order interactions across parameters, having therefore $S_i \approx S_{T_i}$. I.e., most of the variance can be explained with the individual parameters first-order effects. This is not the case for _C_ type functions, for which higher-order interactions play a prominent role and the inequality $S_{T_i} > S_i$ holds.

The functions introduced in equations ([1](#eq:1)),  ([2](#eq:2)), ([3](#eq:3)) and ([4](#eq:4)) can be loaded from an external module. Non-standard modules such as `sobol_seq` and `chaospy` are required. However, they can be easily installed by typing 
        `pip install sobol_seq` 
from your terminal (Linux) or from the anaconda-prompt environment in case you use Anaconda package manager.

```{.python .input}
# import modules
import numpy as np
import pandas as pd
import sobol_seq
from string import ascii_lowercase
from testfunctions import create_dict, functions, AE_dic, AEF_dic
```

The analytical expression of the sensitivity indices can be analytically computed for all the functions and can be adopted for benchmarking the performance of the sensitivity indices computed. <br/>
As regards function $A_1$, 

<!--
Equation labels as ordinary links -->
<div id="eq:5"></div>

$$
\begin{equation}
E_{k2} = \frac{1}{6}*(1-{\frac{1}{6}}^k)+\frac{4}{15}*((-1)^{k+1}*{\frac{1}{2}}^k+{\frac{1}{3}}^k) \
\label{_auto5} \tag{5}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div id="eq:6"></div>

$$
\begin{equation}
V(k) = \frac{1}{10}*{\frac{1}{3}}^k+\frac{1}{18}-{\frac{1}{9}}*({\frac{1}{9}}^{2k})+(-1)^{k+1}*\frac{2}{45}*{\frac{1}{2}}^k) \
\label{_auto6} \tag{6}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div id="eq:7"></div>

$$
\begin{equation}
E_i(i) = \frac{1}{6}*(1-{\frac{1}{3}}^i)+\frac{4}{15}*((-1)^{i+1}*{\frac{1}{2}}^i+{\frac{1}{2}}^i) \
\label{_auto7} \tag{7}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div id="eq:8"></div>

$$
\begin{equation}
T_1 = \frac{1}{2}*{\frac{1}{3}}^{i-1}*(1-{\frac{1}{3}}^{k-i}) \
\label{_auto8} \tag{8}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div id="eq:9"></div>

$$
\begin{equation}
T_2 = \frac{1}{2}*({\frac{1}{3}}^i-{\frac{1}{3}}^k) \
\label{_auto9} \tag{9}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div id="eq:10"></div>

$$
\begin{equation}
T_3 = \frac{3}{5}*(4*{\frac{1}{3}}^{k+1})+(-1)^{i+k+1}*{\frac{1}{2}}^{k-i+2}+{\frac{1}{3}}^{i+1}) \
\label{_auto10} \tag{10}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div id="eq:11"></div>

$$
\begin{equation}
T_4 = \frac{1}{5}*((-1)^{i+2}*\frac{1}{3}*{\frac{1}{2}}^{i-2}-4*{\frac{1}{3}}^{i+1}) \
\label{_auto11} \tag{11}
\end{equation}
$$

<!--
Equation labels as ordinary links -->
<div id="eq:12"></div>

$$
\begin{equation}
T_5 = \frac{1}{5}*((-1)^{k+1}*\frac{1}{3}*{\frac{1}{2}}^{k-2}+(-1)^{k+i}*{\frac{1}{3}}^{i+1}*{\frac{1}{2}}^{k-i-2}) \
\label{_auto12} \tag{12}
\end{equation}
$$

From which one can derive the expression for the total-order sensitivity indices $S_T_i $as:

<!--
Equation labels as ordinary links -->
<div id="eq:13"></div>

$$
\begin{equation}
S_T_i = \frac{E_{k2}(k)-E_i_(i)-/frac{1}{4}*(T_1(i)-2*T_2(i)+T_3(i)-T_4(i)-T_5(I))}{V(k)} \
\label{_auto13} \tag{13}
\end{equation}
$$

As well as first order indices:


<!--
Equation labels as ordinary links -->
<div id="eq:14"></div>

$$
\begin{equation}
S_i = /frac{(E_{k2}(k)-E_i_(i)-/frac{1}{4}*(T_1(i)-2*T_2(i)+T_3(i)-T_4(i)-T_5(I))*(12*(1-{/frac{1}{2}}^k)^2))}{V(k)*27*(/frac{1}{2}-/frac{4}{5}*{/frac{-1}{2}}^k+/frac{3}{10}*/frac{1}{3}^k)} \
\label{_auto14} \tag{14}
\end{equation}
$$

The sensitivity indices $S_i$ and $S_{Ti}$ for $Y=G(X)$ in
eq. ([1](#eq:1)) may
be derived as outlined in [Saltelli 2010](https://www.sciencedirect.com/science/article/pii/S0010465509003087)).
The conditional variance $V_i$ may be found to be:

<!-- Equation labels as ordinary links -->
<div id="eq:3"></div>

$$
\begin{equation}
V_i \left ( G^*(X_i,a_i,\alpha_i) \right) =
\frac{\alpha_i^2}{(1+2\alpha_i)(1+a_i)^2}
\label{eq:3} \tag{3}
\end{equation}
$$

while the $V_{T_I}$ and the variance $V$ are given by the same
expressions as
for the [g_function](interactive_g_function.ipynb).

<!-- Equation labels as ordinary links -->
<div id="eq:4"></div>

$$
\begin{equation}
V_{T_i} = V_i \; \prod_{j\neq i} (1+V_j) \qquad \text{and}
\qquad V = \prod_{i=1}^k (1+V_i) -1
\label{eq:4} \tag{4}
\end{equation}
$$

Consequently the first order sensitivity indices $S_i$ of $Y=G^*(X)$, are given by

<!-- Equation labels as ordinary links -->
<div id="eq:5"></div>

$$
\begin{equation}
S_i=\frac{V_i}{V} \qquad \text{and} \qquad
S_{T_i}=\frac{V_{T_i}}{V}
\label{eq:5} \tag{5}
\end{equation}
$$

<!-- The expressions for the variance obtained when keeping one parameter -->
<!-- fixed and varying all the others can be found below alow with the -->
<!--
expression for the total variance.  The Sensitivity indices -->
<!-- expressions
can be easily retrieved from these. -->


<!-- In the code snippet below alow
you to experiment interactively to so -->
<!-- how the values of $a_i$ affect
the correspoding $S_i$, i.e the -->
<!-- sensitivity of $G$ with respect to
$X_i$. -->

```{.python .input  n=3}
# Analytical computations
f, ax = plt.subplots(1,1)
f.suptitle('G* function with variable coefficients')

# import modules
import numpy as np

def Vi(ai,alphai):
    return alphai**2/((1+2*alphai)*(1+ai)**2)

def V(a_prms,alpha):
    D=1
    for ai,alphai in zip(a_prms,alpha):
        D*=(1+Vi(ai,alphai))     
    return D-1


def S_i(a,alpha):
    S_i=np.zeros_like(a)
    for i, (ai,alphai) in enumerate(zip(a,alpha)):
        S_i[i]=Vi(ai,alphai)/V(a,alpha)
    return S_i

def S_T(a,alpha):
    # to be completed
    S_T=np.zeros_like(a)
    Vtot=V(a,alpha)
    for i, (ai,alphai) in enumerate(zip(a,alpha)):
        S_T[i]=(Vtot+1)/(Vi(ai,alphai)+1)*Vi(ai,alphai)/Vtot
    return S_T


def update_Sobol(**kwargs):
    import re
    r = re.compile("([a-zA-Z]+)([0-9]+)")
    ax.clear()
    prm_cat=int(len(kwargs)/k)
    prms=np.zeros((prm_cat,k))
 
    for key, value in kwargs.items(): #find indx and value for a_prms
        pre,post=r.match(key).groups()
        cat_idx=strings.index(pre)
        prms[cat_idx,int(post)]=value
            
        
    Si[:]=S_i(prms[0,:],prms[1,:])
    ST[:]=S_T(prms[0,:],prms[1,:])
    width=0.4
    x_tick_list=np.arange(len(prms[0,:]))+1
    ax.set_xticks(x_tick_list+width/2)
    x_labels=['x'+str(i) for i in np.arange(len(prms[0,:]))]
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0,1)

    ax.bar(x_tick_list,Si,width,color='blue')
    ax.bar(x_tick_list+width,ST,width,color='red')        
    ax.legend(['First order indices','Total indices'])

k=4 #number of prms
strings=['a','alpha','delta']
a_lbls=[strings[0]+str(i) for i in np.arange(k)]
alpha_lbls=[strings[1]+str(i) for i in np.arange(k)]
delta_lbls=[strings[2]+str(i) for i in np.arange(k)]
Si=np.zeros(k)
ST=np.zeros(k)
a_prms=np.zeros(k)
alpha=np.zeros_like(a_prms)
delta=np.zeros_like(a_prms)



import ipywidgets as widgets    
my_sliders=[]
for i in range(k):
    my_sliders.append(widgets.FloatSlider(min=0, max=15, value=6.52, description=a_lbls[i]))
    my_sliders.append(widgets.FloatSlider(min=0, max=15, value=1.0, description=alpha_lbls[i]))
    my_sliders.append(widgets.FloatSlider(min=0, max=1.0, value=0.5, description=delta_lbls[i]))


slider_dict = {slider.description:slider for slider in my_sliders}
ui_left = widgets.VBox(my_sliders[0::3]) 
ui_mid  = widgets.VBox(my_sliders[1::3])
ui_right = widgets.VBox(my_sliders[2::3])
ui=widgets.HBox([ui_left,ui_mid,ui_right])


out=widgets.interactive_output(update_Sobol, slider_dict) 

display(ui,out)
```

Use the sliders to see how the sensitivities vary with the values of $a_i$, and
reflect on the effect.

Note from the analytical expressions $V_i$ and
$V_{T_i}$ (derived in
[[saltelli2010]](#saltelli2010)) in the sensitivity
indices do not depend on the
shift paramters $\delta_i$.

For $\alpha_i<1$ the
$G^*$ function is concave, while the function is
convex for $\alpha_i>1$, which
is the reason for calling $\alpha$ a
shape parameter.

## Approximation of the sensitivity indices for Sobol's $G^*$ function with spectral expansions

In this
section we show the spectral expansion module
[chaospy](https://github.com/jonathf/chaospy) may be used to compute the Sobol
indices for Sobol's $G^*$ function.  A more in depth treatment of
`chaospy` and
its usage is provided in the separate notebook [A
practical introduction to
polynomial chaos with the chaospy package](introduction_gpc.ipynb). Furthermore,
you may find our previous "A
Guide to Uncertainty Quantification and Sensitivity
Analysis for
Cardiovascular Applications" [Eck et al. 2015](https://onlinelibrary.wiley.com/doi/full/10.1002/cnm.2755) as a
useful
introduction to how polynomial chaos expansions may be used for
UQ&S. We
are therefore focusing on the application of the spectral
expansions and how
they agree with the analytical solutions for the
indices, rather than presenting
the spectral expansion theory.

```{.python .input}
# chaospy G-function with sliders
import chaospy as cp

if not 'jpdf' in globals():
    jpdf = cp.Iid(cp.Uniform(),k) #the joint pdf
    print('Create the joint pdf')


def update_chaospy_G(**kwargs):
    NS=kwargs['NS']
    del kwargs['NS']
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


    X=jpdf.sample(NS)
    print('Number of samples: ',NS)

    G_sample=G(X.transpose(),prms[0,:],prms[1,:],prms[2,:])

    poly = cp.orth_ttr(polynomial_order, jpdf)
    approx = cp.fit_regression(poly, X, G_sample)

    exp_pc = cp.E(approx, jpdf)
    std_pc = cp.Std(approx, jpdf)
    print("Statistics polynomial chaos\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('pc  : {:2.5f} | {:2.5f}'.format(float(exp_pc), std_pc))
    S_pc = cp.Sens_m(approx, jpdf) #Si from chaospy
    S_tpc = cp.Sens_t(approx, jpdf) #Total effect sensitivity index from chaospy
    
    row_labels= ['S_'+str(idx) for idx in range(len(a_prms))]
    col_labels=['Chaospy','Analytical','Error (%)']

    print("\nFirst Order Indices")
    print_vectors_relerror(S_pc,Si,col_labels,row_labels,[3,3,0])

    print("\n\nTotal Effect Indices")
    row_labels= ['St_'+str(idx) for idx in range(k)]
    print_vectors_relerror(S_tpc,ST,col_labels,row_labels,[3,3,0])


if (len(my_sliders)==len(a_prms)*3):   #add sliders if not added before
    my_sliders.append(widgets.IntSlider(min=500,max=5000,step=200,value=500,description='NS')) #add slider for samples
    my_sliders.append(widgets.IntSlider(description='polynomial_order', min=1,max=6,value=4)) # add slider for polynomial order

    slider_dict = {slider.description:slider for slider in my_sliders} #add the sliders in the dictionary 

    ui_left = widgets.VBox(my_sliders[0::3]) 
    ui_mid  = widgets.VBox(my_sliders[1::3])
    ui_right = widgets.VBox(my_sliders[2::3])
    ui=widgets.HBox([ui_left,ui_mid,ui_right])

out=widgets.interactive_output(update_chaospy_G, slider_dict) 
display(ui,out)

# end chaospy G-function with sliders
```

You may check whether approximated sensitivity indices are independent of the
shift paramters $\delta_i$, as they should according according to the analytical
expressions for $V_i$ in eq. ([3](#eq:3)).

## Comparison of MC and PC approximation of the sensitivity indices

In this section we demonstrate how
Monte Carlo simulations and
polynomial chaos expansions may be used to
estimate the Sobol
indices and compare their estimates with the analytical
solutions.

Readers unfamiliar with how to use python (notebooks) for Monte
Carlo
simulations and polynomial chaos expansions are referred to our
previous
[A brief introduction to UQ and SA with the Monte Carlo
method](https://github.com/lrhgit/uqsa_tutorials/blob/master/monte_carlo.ipynb).
and  [A practical introduction to polynomial chaos with the
chaospy
package](https://github.com/lrhgit/uqsa_tutorials/blob/master/introduction_gpc.ipynb)

```{.python .input}
# mc and pc comparison for Gstar-function with sliders

import monte_carlo as mc

if not 'jpdf' in globals():
    jpdf = cp.Iid(cp.Uniform(),k) #the joint pdf
    print('Create the joint pdf')


def update_Gstar(**kwargs):
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
        
    ## Update the analytical indices
    Si[:]=S_i(prms[0,:],prms[1,:])
    ST[:]=S_T(prms[0,:],prms[1,:])


    ## Monte Carlo update
    print('Number of samples for Monte Carlo: ', NsMC) 
    X_mc=jpdf.sample(NsMC)
    A, B, C = mc.generate_sample_matrices_mc(NsMC, k, jpdf, sample_method='R') #A, B, C already transposed
    G_A_sample = G(A,prms[0,:],prms[1,:],prms[2,:])
    G_B_sample = G(B,prms[0,:],prms[1,:],prms[2,:])
    G_C_sample_list = np.array([G(C_i,prms[0,:],prms[1,:],prms[2,:]) for C_i in C]).T
    
    exp_mc = np.mean(G_A_sample)
    std_mc = np.std(G_A_sample)
    print("Statistics Monte Carlo\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('mc  : {:2.5f} | {:2.5f}'.format(float(exp_mc), std_mc))
    
    S_mc, S_tmc = mc.calculate_sensitivity_indices_mc(G_A_sample, G_B_sample, G_C_sample_list)


    ## update PC estimates
    Xpc=jpdf.sample(NsPC)
    print('Number of samples: ',NsPC)

    G_sample=G(Xpc.transpose(),prms[0,:],prms[1,:],prms[2,:])

    poly = cp.orth_ttr(polynomial_order, jpdf)
    approx = cp.fit_regression(poly, Xpc, G_sample)

    exp_pc = cp.E(approx, jpdf)
    std_pc = cp.Std(approx, jpdf)
    print("Statistics polynomial chaos\n")
    print('\n        E(Y)  |  std(Y) \n')
    print('pc  : {:2.5f} | {:2.5f}'.format(float(exp_pc), std_pc))
    S_pc = cp.Sens_m(approx, jpdf) #Si from chaospy
    S_tpc = cp.Sens_t(approx, jpdf) #Total effect sensitivity index from chaospy
    
    row_labels= ['S_'+str(idx) for idx in range(len(a_prms))]
    #col_labels=['Chaospy','Analytical','Error (%)']
    col_labels=['Monte Carlo','Err (%)','PolyChaos','Err (%)']


    print("\nFirst Order Indices")
#    print_vectors_relerror(S_pc,Si,col_labels,row_labels,[3,3,0])
    print_3vectors_relerror(S_mc,S_pc, Si, col_labels, row_labels, [3,0,3,0])

    print("\n\nTotal Effect Indices")
    row_labels= ['St_'+str(idx) for idx in range(k)]
#    print_vectors_relerror(S_tpc,ST,col_labels,row_labels,[3,3,0])
    print_3vectors_relerror(S_tmc,S_tpc, ST, col_labels, row_labels, [3,0,3,0])



## Set up the sliders 
cmp_sliders=[]
for i in range(k):
    cmp_sliders.append(widgets.FloatSlider(min=0, max=15, value=6.52, description=a_lbls[i]))
    cmp_sliders.append(widgets.FloatSlider(min=0, max=15, value=1.0, description=alpha_lbls[i]))
    cmp_sliders.append(widgets.FloatSlider(min=0, max=1.0, value=0.5, description=delta_lbls[i]))

cmp_sliders.append(widgets.IntSlider(min=500,max=5000,step=200,value=500,description='NsPC')) #add slider for samples
cmp_sliders.append(widgets.IntSlider(min=500,max=50000,step=200,value=500,description='NsMC')) #add slider for samples
cmp_sliders.append(widgets.IntSlider(description='polynomial_order', min=1,max=6,value=4)) # add slider for polynomial order

slider_dict = {slider.description:slider for slider in cmp_sliders} #add the sliders in the dictionary 

ui_left = widgets.VBox(cmp_sliders[0::3]) 
ui_mid  = widgets.VBox(cmp_sliders[1::3])
ui_right = widgets.VBox(cmp_sliders[2::3])
ui=widgets.HBox([ui_left,ui_mid,ui_right])

out=widgets.interactive_output(update_Gstar, slider_dict) 
display(ui,out)
```
