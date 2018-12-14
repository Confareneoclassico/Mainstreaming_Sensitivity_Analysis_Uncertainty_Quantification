# Sensitivity indices for Sobol's $G^{*}$ function

<!-- dom:AUTHOR: Leif Rune
Hellevik -->
<!-- Author: -->  
**Leif Rune Hellevik**, [leif.r.hellevik@ntnu.no](mailto:leif.r.hellevik@ntnu.no)

# Sobol's $G^{*}$function
<div id="sec:G_functions"></div>

In our previous notebook
[g_function](interactive_g_function.ipynb),
we demonstrated how polynomial chaos
expansions (with `chaospy`) and
Monte Carlo methods may be used to approximate
the Sobol sensitivity
indices. The example was Sobol's G function and was taken
from
[Saltelli et al. 2010](https://www.sciencedirect.com/science/article/pii/S0010465509003087)
too.

In this notebook we focus on another function, which has proved to be
useful as a test function with analytical
solutions for the sensitivity
indices, namely Sobol's $G^*$ function which
is defined much in the same manner
as the [g_function](interactive_g_function.ipynb):

<!-- Equation labels as ordinary links -->
<div id="eq:1"></div>

$$
\begin{equation}
Y=G(X) =  G(X_1, X_2,\ldots,X_k,a_1, a_2,\ldots,a_k)  =
\prod_{i=1}^{k} g_i \label{eq:1} \tag{1}
\end{equation}
$$

which is identical with

<!-- Equation labels as ordinary links -->
<div id="eq:2"></div>

$$
\begin{equation}
g_i = \frac{(1+\alpha_i) |2 \left (X_i+ \delta_i -
I(X_i+\delta_i) \right ) -1 |^{\alpha_i}+a_i}{1+{a}_i} \label{eq:2} \tag{2}
\end{equation}
$$

All the input factors $X_i$ are assumed to be uniformly
distributed in the
interval $[0,1]$, an the coefficients $a_i$ are
assumed to be positive real
numbers $(a_i \leq 0)$, $\delta_i \in
[0,1]$, and $\alpha_i >0$. Finally, $
I(X_i+\delta_i)$ denotes the
integer value for $X_i+\delta_i$. Note that for for
$\alpha_i=1$ and
$\delta_i=0$ $g^*$ reduces to $g$ in the
[g_function](interactive_g_function.ipynb) notebook. The $\alpha_i$
and
$\delta_i$ are curvature and shift parameters, respectively.

The number of
factors *k* can be varied as the reader pleases, but the
minimum number to
produce a meaningful inference is set at three.

Run the first cell to
initialise plotting and printing modules for later use (and system
settings)

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

import matplotlib.pyplot as plt
from present_output import print_vectors_relerror, print_3vectors_relerror
```

As you will be able to explore below, the sensitivity $S_i$ of $G$ in
([1](#eq:1)) with respect to a specific input factor $X_i$, will depend
on the
value of the corresponding coefficient $a_i$; small values of
$a_i$ (e.g.
$a_i=0$) will yield a high corresponding $S_i$, meaning
that $X_i$ is an
important/influential variable on the variance or
uncertainty of $G$.

We have
implemented Sobol's  $G^*$ function in ([1](#eq:1)) and  ([2](#eq:2)) in the
code snippet below:

```{.python .input  n=2}
# model function
import numpy as np
from numba import jit

@jit
def g(Xj,aj,alphaj,deltaj):
    return ((1+alphaj)*np.abs(2*(Xj+deltaj-(Xj+deltaj).astype(int))-1)**alphaj+aj)/(1+aj)


@jit
def G(X,a,alpha,d):
    G_vector=np.ones(X.shape[0])

    for j, aj in enumerate(a):
        np.multiply(G_vector,g(X[:,j],aj,alpha[j],d[j]),G_vector)
    return G_vector
```

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
    my_sliders.append(widgets.IntSlider(min=500,max=5000,step=250,value=500,description='NS')) #add slider for samples
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

cmp_sliders.append(widgets.IntSlider(min=500,max=5000,step=250,value=500,description='NsPC')) #add slider for samples
cmp_sliders.append(widgets.IntSlider(min=500,max=50000,step=250,value=500,description='NsMC')) #add slider for samples
cmp_sliders.append(widgets.IntSlider(description='polynomial_order', min=1,max=6,value=4)) # add slider for polynomial order

slider_dict = {slider.description:slider for slider in cmp_sliders} #add the sliders in the dictionary 

ui_left = widgets.VBox(cmp_sliders[0::3]) 
ui_mid  = widgets.VBox(cmp_sliders[1::3])
ui_right = widgets.VBox(cmp_sliders[2::3])
ui=widgets.HBox([ui_left,ui_mid,ui_right])

out=widgets.interactive_output(update_Gstar, slider_dict) 
display(ui,out)
```
