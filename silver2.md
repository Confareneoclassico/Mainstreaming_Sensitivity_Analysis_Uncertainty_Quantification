# Silver as constraint to large scale PV development

<!-- AUTHOR: Samuele lo
Piano -->

**Samuele Lo Piano** [slopiano@gmail.com](mailto:s.lopiano@gmail.com)
In this example we will analyse whether silver could constrain a large-scale
development of solar photovoltaics. Silver is used in the contact metallization
of crystalline-silicon-based PV cells. Remarkable efforts are being allocated
towards silver-paste-use reduction. On the other hand, solar photovoltaics
modules are being installed at an exponential increasing pace. Other parameters
are also included in our model, i.e.:

* other (non-industrial and industrial non-PV) uses of silver

* increase in efficiency of the modules

We will perform
a sensitivity analysis of a model aimed at defining silver extraction capacity
and the compatibility with its biophysical budget up to the year 2050.

Will
silver use in photovoltaics be the most pressing constraint or would other
sector impose more severe burdens? How about the extraction capacity - on the
base of the silver mining trends, could silver mining respond to a significant
uprise in silver demand? And finally, how about silver natural budget, can we be
sure the expansion of photovoltaics will not be constrained by silver natural
availability?

Let's start off by setting the plot environment and importing the
relevant packages.

```{.python .input  n=1}
# ipython magic
%matplotlib notebook
%load_ext autoreload
%autoreload 2
```

```{.python .input  n=2}
%matplotlib inline

# plot configuration
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use("ggplot")
# import seaborn as sns # sets another style
matplotlib.rcParams['lines.linewidth'] = 3
fig_width, fig_height = (7.0,5.0)
matplotlib.rcParams['figure.figsize'] = (fig_width, fig_height)

# font = {'family' : 'sans-serif',
#         'weight' : 'normal',
#         'size'   : 18.0}
# matplotlib.rc('font', **font)  # pass in the font dict as kwar
```

```{.python .input  n=3}
# import modules
import numpy as np
from scipy import special
import sobol_seq
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
```

The yearly silver demand can be described as such:

<!-- Equation labels as
ordinary links -->
<div id="_auto1"></div>

$$
\begin{equation}
m_{Ag,yearly}(t)
=  \frac{m_{Ag,PV} e^{-Ag_{contact paste}\,t} PC_{PV} e^{-PC_{PVinstall}\,t}}{PV_{efficiency} e^{cell_{efficiency}\,t}}\  + Ag_{nonPV}(t)\
\label{_auto1} \tag{1}
\end{equation}
$$

Where $t$ is time relative to the base
year (2017); $m_{Ag,PV}$ is the amount of silver paste used per unit of
photovoltaic cell (100 mg/cell) in the starting year (2017); $PV_{PC}$ is the
yearly amount of first-generation crystalline-silicon-based power capacity
installed in 2017 (74 GW); $Ag_{nonPV}$ is the amount of silver used outside of
the solar photovoltaic sector; $PV_{efficiency}$ represents the typical cell
efficiency (4.27 W/cell, corresponding to 17.5%) in the base year. The typical
loss in efficiency from cells to modules has been considered as insignificant
for the purposes of the present accounting.

## Definining the variables and their distributions

The distributions of our parameters have been defined from
uniform Sobol distributions, which have been conveniently re-scaled or
transformed to other size and distribution shapes.

```{.python .input  n=4}
# Sample basis distribution
run = 50

sampleTot = sobol_seq.i4_sobol_generate(12,run*2**10)
```

The pace at which the silver-paste use for the contact metallization of solar
photovoltaic cells has been defined through a stakeholder-
engagment exercise.
Field experts have been contacted and have been asked to produce a guesstimate
on the silver use for contact metallization to the year 2050. These figures have
been complemented with a report, whose findings have been extrapolated to the
year 2050 - [ITRPV2018](http://www.itrpv.net/Reports/Downloads/).

```{.python .input}
# Current use of silver paste (mg/cell) - year 2017

ITRPV2018silverCell2017 = 100
ITRPVsilverCell2028 = 50
ITRPV2018decreasePace= - 1/(2028-2017) * np.log(ITRPVsilverCell2028/ITRPV2018silverCell2017)

def decreasePaceSilver2050(silverCell):
    return - 1/(2050-2017) * np.log(silverCell/ITRPV2018silverCell2017)

def mu(x):
    return np.mean(x)

def sigma(x):
    return np.std(x)
```

The upper boundary is conversely defined from the higher figure of the
distribution. In order to generate this latter, the figures provided have been
weighted according to the frequency of the responses received. A sample of
quasi-random numbers is eventually obtained from the truncated normal
distribution. A convenient size is chosen, i.e. $2^{10}$ - more than 1,000
values. The reader is encouraged to evaluate the effect of the sample size on
the basis of what has been illustrated for the convergence pace of the test
functions in [another notebook of this series](testfunctions3.ipynb). Here the
effect of the variability of the sample has been averaged out by repeating the
simulation 50 times.

```{.python .input}
silverCell2050DecreasePace = [ITRPV2018decreasePace, ITRPV2018decreasePace, \
decreasePaceSilver2050(ITRPV2018silverCell2017*0.01), decreasePaceSilver2050(ITRPV2018silverCell2017*0.01),\
decreasePaceSilver2050(10), decreasePaceSilver2050(15),decreasePaceSilver2050((20+10)/2), 0.05]
```

The growth in photovoltaic-cell efficiency has been taken as deterministic in
this study.

```{.python .input}
# PV Cell Efficiency growth up to the year 2050 (W/cell)

PVCellEfficiency2017 = 4.27

PVCellEfficiency2050 = 6.1

PVCellEfficiencyIncreasePace = 1/(2050-2017) * np.log(PVCellEfficiency2050/PVCellEfficiency2017)
```

Analogously, the distribution for the trend in the installed PV power capacity
is based on reports produced by energy agencies and research institutes (IEA,
Greenpeace and the Fraunhofer institute). It has been assumed the market share
of silicon-based PV solar panels would remain constant at 90%.

```{.python .input}
# Current expansion of PV power capacity (GW) - year 2017

PVPC2017 = 82
CrystallineSiliconShare = 0.9
IEA2050 = 4670
Greenpeace2050 = 9295
Fraunhofer2050 = 30700
CPVPC2017 = 402

IEA2050_CrystallineSiliconShare = IEA2050 * CrystallineSiliconShare
Greenpeace2050_CrystallineSiliconShare = Greenpeace2050 * CrystallineSiliconShare
Fraunhofer2050_CrystallineSiliconShare = Fraunhofer2050 * CrystallineSiliconShare
PVPC2017_CrystallineSiliconShare = PVPC2017 * CrystallineSiliconShare

PVPC2050IncreasePace = [0.025, 0.112, 0.061]
```

From which a truncated normal distribution has been generated. The low-
discrepancy Sobol sequence can be converted into other shape transformation by
evaluating the inverse probability distribution. As the value of probability
distribution shall be the same in order to maintain the low-discrepancy
properties of the Sobol sequence, one can apply an inverse transformation to
assure consistency. 

The figures have been defined on the basis of the
[Thompson-Reuter/Silver Institute report](http://www.silverinstitute.org/wp-
content/uploads/2017/08/WSS2017.pdf) as regards the trends of silver use in
other industrial application and non-PV industrial applications. Their future
projections are based on the autoregressive integrated moving average
[ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) approach.

```{.python .input}
# Non-industrial + Industrial Non-PV
otherAg = pd.DataFrame([28.8,29.4,32.8,26.0,30.7,32.5,29.3,33.4,33.2,34.0,29.9,28.7], index=[y for y in range(2006,2018)]).T
```

The trend for the yearly extraction capacity has been retrieve from the USGS
figures and is based on the projection from post World-War-II onwards.

```{.python .input}
# Yearly extraction capacity

Extraction = pd.DataFrame([3.97,5.22,5.44,5.57,6.32,6.21,6.7,6.9,6.67,7,7.02,7.19,7.43,6.91,7.32,7.37,7.65,7.78,7.73,8.01,8.3,8.03,8.56,9.2,9.36,9.17,9.38,9.7,9.26,9.43,9.84,10.3,
10.7,10.8,10.7,11.2,11.5,12.1,13.1,13.1,13,14,15.5,16.4,16.6,15.6,14.9,14.1,14,14.9,15.1,16.5,17.2,17.6,18.1,18.7,18.8,18.8,20,20.8,20.1,20.8,21.3,22.3,23.3,23.3,24.3,25.7,
26.8,27.6,25.7,25], index=[y for y in range(1946,2018)],columns=['Global Production'])
```

As regards the biophysical available budget, the figures for the ultimate
recoverable resources have been obtained by the literature. A uniform
distribution has been selected also in this case, whose extremes are the most
conservative and the most optmistic figures available in the literature.

### Output functions

We are now ready to generate the output functions: i) the
demand from the photovoltaics sector is evaluated along ii) the overall silver
demand and iii) its cumulative figure up to the year 2050. iv) The derivative of
the yearly demand has also been computed. In order to analyse any viability
constraint (technical incapacity to mine adequate supply of minerals to face the
demand on a year-to-year base) and any potential feasibility constraint
(availability of silver natural budget), the trends of these variables have also
been assessed below. As regards the former, it is assumed silver from old-scrap
recycling provides only 20% of the yearly supply. This figure is in line with
the recent global trends reported. The latter is evaluated by comparing the
figures for the cumulative extraction against silver natural budget.

```{.python .input}
run = 50 

S_silverCellR = []
S_PVPCR = []
S_otherAgR = []
T_silverCellR = []
T_PVPCR = []
T_otherAgR = []
yearlySilverR = []
CumulativeSilverR = []
PVPC2050R = []
silverCellR = []
otherAgR = []
silverDeltaR = []
DifferenceR = []

for r in range(run):
    sample = sampleTot[int(r*len(sampleTot)/run):int((r+1)*len(sampleTot)/run)].T

    silverCellDistribution2 = sigma(silverCell2050DecreasePace)*2**0.5*special.erfinv(sample[0]*(1-special.erf(-mu(silverCell2050DecreasePace)/
    (sigma(silverCell2050DecreasePace)*2**0.5)))+special.erf(-mu(silverCell2050DecreasePace)/
    (sigma(silverCell2050DecreasePace)*2**0.5)))+mu(silverCell2050DecreasePace)

    silverCellDistribution2B = sigma(silverCell2050DecreasePace)*2**0.5*special.erfinv(sample[6]*(1-special.erf(-mu(silverCell2050DecreasePace)/
    (sigma(silverCell2050DecreasePace)*2**0.5)))+special.erf(-mu(silverCell2050DecreasePace)/
    (sigma(silverCell2050DecreasePace)*2**0.5)))+mu(silverCell2050DecreasePace)

    PVPC2050IncreasePaceDist = sigma(PVPC2050IncreasePace)*2**0.5*special.erfinv(sample[1]*(1-special.erf(-mu(PVPC2050IncreasePace)/
    (sigma(PVPC2050IncreasePace)*2**0.5)))+special.erf(-mu(PVPC2050IncreasePace)/
    (sigma(PVPC2050IncreasePace)*2**0.5)))+mu(PVPC2050IncreasePace)

    PVPC2050IncreasePaceDistB = sigma(PVPC2050IncreasePace)*2**0.5*special.erfinv(sample[7]*(1-special.erf(-mu(PVPC2050IncreasePace)/
    (sigma(PVPC2050IncreasePace)*2**0.5)))+special.erf(-mu(PVPC2050IncreasePace)/
    (sigma(PVPC2050IncreasePace)*2**0.5)))+mu(PVPC2050IncreasePace)

    Noise = sigma(sample[2])*2**0.5*special.erfinv(2*sample[2]-1)
    NoiseB = sigma(sample[8])*2**0.5*special.erfinv(2*sample[8]-1)

    # fit model Silver
    model = ARIMA(otherAg.iloc[-1]+Noise[:len(otherAg.T)], order=(0, 1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(otherAg.iloc[-1]), 44, typ='levels')
    yhat.index = yhat.index + 2006
    forecast, stderr, conf = model_fit.forecast(33)

    modelB = ARIMA(otherAg.iloc[-1]+NoiseB[:len(otherAg.T)], order=(0, 1, 1))
    model_fitB = modelB.fit(disp=False)
    # make prediction
    yhatB = model_fitB.predict(len(otherAg.iloc[-1]), 44, typ='levels')
    yhatB.index = yhat.index
    forecastB, stderrB, confB = model_fitB.forecast(33)

    otherAgDist = pd.DataFrame([((forecast[i]-conf[i][0])/1.96)*2**0.5*special.erfinv(sample[3]*(1-special.erf(-forecast[i]/
    (((forecast[i]-conf[i][0])/1.96)*2**0.5)))+special.erf(-forecast[i]/
    (((forecast[i]-conf[i][0])/1.96)*2**0.5)))+forecast[i] for i in range(len(forecast))],index=[i for i in range(2018,2051)])

    otherAgDistB = pd.DataFrame([((forecastB[i]-confB[i][0])/1.96)*2**0.5*special.erfinv(sample[9]*(1-special.erf(-forecastB[i]/
    (((forecastB[i]-confB[i][0])/1.96)*2**0.5)))+special.erf(-forecastB[i]/
    (((forecastB[i]-confB[i][0])/1.96)*2**0.5)))+forecastB[i] for i in range(len(forecastB))],index=[i for i in range(2018,2051)])

    # Resources and reserves
    distributionSilverReserveResource = sample[4]*(4000-530)+530
    distributionSilverReserveResource = sample[10]*(4000-530)+530

    # fit model Extraction
    model2 = ARIMA(Extraction['Global Production']+Noise[:len(Extraction)], order=(2, 1, 1))
    model_fit2 = model2.fit(disp=False)
    # make prediction
    yhat2 = model_fit2.predict(len(Extraction), 105, typ='levels')
    yhat2.index = yhat2.index+1946
    forecast2, stderr2, conf2 = model_fit2.forecast(33)

    model2B = ARIMA(Extraction['Global Production']+NoiseB[:len(Extraction)], order=(2, 1, 1))
    model_fit2B = model2B.fit(disp=False)
    # make prediction
    yhat2B = model_fit2B.predict(len(Extraction), 105, typ='levels')
    yhat2B.index = yhat2B.index+1946
    forecast2B, stderr2B, conf2B = model_fit2B.forecast(33)

    # Extraction distribution
    ExtractionDist = pd.DataFrame([((forecast2[i]-conf2[i][0])/1.96)*2**0.5*special.erfinv(sample[5]*(1-special.erf(-forecast2[i]/
    (((forecast2[i]-conf2[i][0])/1.96)*2**0.5)))+special.erf(-forecast2[i]/
    (((forecast2[i]-conf2[i][0])/1.96)*2**0.5)))+forecast2[i] for i in range(len(forecast2))],index = otherAgDist.index)

    ExtractionDistB = pd.DataFrame([((forecast2B[i]-conf2B[i][0])/1.96)*2**0.5*special.erfinv(sample[11]*(1-special.erf(-forecast2B[i]/
    (((forecast2B[i]-conf2B[i][0])/1.96)*2**0.5)))+special.erf(-forecast2B[i]/
    (((forecast2B[i]-conf2B[i][0])/1.96)*2**0.5)))+forecast2B[i] for i in range(len(forecast2B))],index = otherAgDistB.index)

    # Generation of the yearly silver output (1,000 metric tonnes)
    PVDemand = pd.DataFrame([(ITRPV2018silverCell2017 * np.exp(-silverCellDistribution2*i1)/(PVCellEfficiency2017 * \
    np.exp(PVCellEfficiencyIncreasePace*i1)))* PVPC2017 * CrystallineSiliconShare * np.exp(PVPC2050IncreasePaceDist*i1)/1000 for i1
    in range (2051-2018)],index=otherAgDist.index)

    PVDemandAB1 = pd.DataFrame([(ITRPV2018silverCell2017 * np.exp(-silverCellDistribution2B*i1)/(PVCellEfficiency2017 * \
    np.exp(PVCellEfficiencyIncreasePace*i1)))* PVPC2017 * CrystallineSiliconShare * np.exp(PVPC2050IncreasePaceDist*i1)/1000 for i1
    in range (2051-2018)],index=otherAgDist.index)

    PVDemandAB2 = pd.DataFrame([(ITRPV2018silverCell2017 * np.exp(-silverCellDistribution2*i1)/(PVCellEfficiency2017 * \
    np.exp(PVCellEfficiencyIncreasePace*i1)))* PVPC2017 * CrystallineSiliconShare * np.exp(PVPC2050IncreasePaceDistB*i1)/1000 for i1
    in range (2051-2018)],index=otherAgDist.index)

    PVDemandB = pd.DataFrame([(ITRPV2018silverCell2017 * np.exp(-silverCellDistribution2B*i1)/(PVCellEfficiency2017 * \
    np.exp(PVCellEfficiencyIncreasePace*i1)))* PVPC2017 * CrystallineSiliconShare * np.exp(PVPC2050IncreasePaceDistB*i1)/1000 for i1
    in range (2051-2018)],index=otherAgDist.index)

    # Generation of the cumulative silver output (1,000 metric tonnes)
    yearlySilver = PVDemand+otherAgDist
    yearlySilverAB1 = PVDemandAB1+otherAgDist
    yearlySilverAB2 = PVDemandAB2+otherAgDist
    yearlySilverAB3 = PVDemand+otherAgDistB
    yearlySilverB = PVDemandB+otherAgDistB

    CumulativeSilver = yearlySilver.cumsum()

    # Derivative Yearly Silver Demand
    derivativeYearlySilver = yearlySilver.diff()

    # Yearly Delta (viability, technological constraint)

    silverDelta = ExtractionDist - 0.8*yearlySilver

    # Cumulative Delta (feasibility, biophysical constraint)

    Difference = distributionSilverReserveResource-CumulativeSilver

    S_silverCell = 1-(0.5*((yearlySilverB-yearlySilverAB1)**2).mean(axis=1))/yearlySilverB.var(axis=1,ddof=0)
    S_PVPC = 1-(0.5*((yearlySilverB-yearlySilverAB2)**2).mean(axis=1))/yearlySilverB.var(axis=1,ddof=0)
    S_otherAg = 1-(0.5*((yearlySilverB-yearlySilverAB3)**2).mean(axis=1))/yearlySilverB.var(axis=1,ddof=0)
    T_silverCell = (0.5*((yearlySilver-yearlySilverAB1)**2).mean(axis=1))/yearlySilver.var(axis=1,ddof=0)
    T_PVPC = (0.5*((yearlySilver-yearlySilverAB2)**2).mean(axis=1))/yearlySilver.var(axis=1,ddof=0)
    T_otherAg = (0.5*((yearlySilver-yearlySilverAB3)**2).mean(axis=1))/yearlySilver.var(axis=1,ddof=0)

    S_silverCellR.append(S_silverCell)
    S_PVPCR.append(S_PVPC)
    S_otherAgR.append(S_otherAg)
    T_silverCellR.append(T_silverCell)
    T_PVPCR.append(T_PVPC)
    T_otherAgR.append(T_otherAg)
    yearlySilverR.append(yearlySilver)
    PVPC2050R.append(PVPC2050IncreasePaceDist)
    silverCellR.append(silverCellDistribution2)
    CumulativeSilverR.append(CumulativeSilver)
    otherAgR.append(otherAgDist)
    silverDeltaR.append(silverDelta)
    DifferenceR.append(Difference)

S_silverCell_df = pd.DataFrame(S_silverCellR)
S_PVPC_df = pd.DataFrame(S_PVPCR)
S_otherAg_df = pd.DataFrame(S_otherAgR)
T_silverCell_df = pd.DataFrame(T_silverCellR)
T_PVPC_df = pd.DataFrame(T_PVPCR)
T_otherAg_df = pd.DataFrame(T_otherAgR)
```

```{.python .input}
df20503d = pd.concat([pd.Series(np.array(silverCellR).flatten()),pd.Series(np.array(PVPC2050R).flatten()),
pd.concat([oa.loc[2050] for oa in otherAgR],axis=0).reset_index(drop=True),                      
pd.concat([yr.loc[2050] for yr in yearlySilverR],axis=0).reset_index(drop=True),
pd.concat([cr.loc[2050] for cr in CumulativeSilverR],axis=0).reset_index(drop=True),
pd.concat([sd.loc[2050] for sd in silverDeltaR],axis=0).reset_index(drop=True),
pd.concat([df.loc[2050] for df in DifferenceR],axis=0).reset_index(drop=True)],axis=1)
df20503d.columns = ['silverCell','PVPC','nonPVsilver','yearlySilver','CumulativeSilver','yearly Delta','Difference']
```

Each of the distributions generated can be visually inspected to acknowledge its
trend in the inquired timeframe with an **uncertainty analysis**.

```{.python .input}
for i2, row in PVDemand.iterrows():
    plt.plot([i2 for col in range(len(PVDemand.columns))], row, c = 'y', label = 'Silver PV Demand' if i2 == 2018 else "")
plt.legend()
plt.show()
```

```{.python .input}
for i2, row in yearlySilver.iterrows():
    plt.plot([i2 for col in range(len(yearlySilver.columns))], row, c = 'r', label = 'Silver Total Demand' if i2 == 2018 else "")
plt.legend()
plt.show()
```

```{.python .input}
for i2, row in CumulativeSilver.iterrows():
    plt.plot([i2 for col in range(len(CumulativeSilver.columns))], row, c = 'g', label = 'Cumulative Silver Demand' if i2 == 2018 else "")
    plt.scatter(i2, row.mean(), c = 'm', label = 'Cumulative Silver demand mean' if i2 == 2018 else "")
plt.legend()
plt.show()
```

```{.python .input}
for i2, row in derivativeYearlySilver.iterrows():
    plt.plot([i2 for col in range(len(derivativeYearlySilver.columns))], row, c = 'k', label = 'derivative Silver' if i2 == 2018 else "")
plt.legend()
plt.show()
```

The trends of viability and feasibility can also be visually inspected.

```{.python .input}
for i2, row in silverDelta.iterrows():
    plt.plot([i2 for col in range(len(silverDelta.columns))], row, c = 'c')
plt.show()
```

```{.python .input}
for i2, row in Difference.iterrows():
    plt.plot([i2 for col in range(len(Difference.columns))], row, c = 'w', label = 'Difference' if i2 == 2018 else "")
    plt.scatter(i2, row.mean(), c = 'm', label = 'Difference mean' if i2 == 2018 else "")
plt.legend()
plt.show()
```

```{.python .input}
for i, row in silverDelta.iterrows():
    plt.bar(i,row[row>0].count(),color='b')
    plt.bar(i,row[row<0].count(),bottom=row[row>0].count(),color='r')
plt.show()
```

```{.python .input}
for i, row in Difference.iterrows():
    plt.bar(i,row[row>0].count(),color='b')
    plt.bar(i,row[row<0].count(),bottom=row[row>0].count(),color='r')
plt.show()
```

One can see that for some combination of values, the expansion of photovoltaics
power capacity may not be feasible or viable. Precisely, constraints may appear
from 2035 onwards. Viability issues seem more likely.

## Sensitivity analysis
What are the parameters whose uncertainty affects the output the most? This
question is of primary importance if one wants to address where the potential
constraint to large-scale development of solar photovoltaics would most likely
come from. To this end, sensitivity analysis can play a role by helping us to
assess which parameters uncertainty will affect the yearly silver demand the
most and with it all the other derived variables.

We will take advantage of
three toolkits to apportion the uncertainty in the output to the input
parameters uncertainty:

i) scatter plots for a rough visual correlation;

ii) Smirnov plots for a more quantitative visual inference;

iii) Variance-based sensitivity indices.

### Scatter plots

These visualizations allow to evaluate
how the uncertainty in the output varies along with the value of the input
variables. These plots can be interpreted in this way: How much may the output
variance be reduced by fixing the input? Is an overall trend visible or only a
random pattern? The input parameter's uncertainty plays a tangible role on the
output's uncertainty if one can discern a trend. Linear regressions have also
been drawn on the scatter plots to enhance the visibility of these effects. The
trends suggests an increasing importance over the course of the years for the PV
variables and the reverse for non-PV silver.

```{.python .input}
for i2, row in yearlySilver.iterrows():
    plt.scatter(silverCellDistribution2, row, s = 2, label = 'silverCellDistribution'+str(i2))
    plt.plot(np.unique(silverCellDistribution2), np.poly1d(np.polyfit(silverCellDistribution2, row, 1))(np.unique(silverCellDistribution2)), color='r')
    plt.legend()
    plt.show()
```

```{.python .input}
for i2, row in yearlySilver.iterrows():
    plt.scatter(silverCellDistribution2, row, s = 2, label = 'silverCellDistribution'+str(i2))
    plt.plot(np.unique(silverCellDistribution2), np.poly1d(np.polyfit(silverCellDistribution2, row, 1))(np.unique(silverCellDistribution2)), color='r')
    plt.legend()
    plt.show()
```

```{.python .input}
for i2, row in yearlySilver.iterrows():
    plt.scatter(otherAgDist.loc[i2], row, s = 2, label = 'Ag_nonPV'+str(i2))
    plt.plot(np.unique(otherAgDist.loc[i2]), np.poly1d(np.polyfit(otherAgDist.loc[i2], row, 1))(np.unique(otherAgDist.loc[i2])), color='b')
    plt.legend()
    plt.show()
```

### Kolmogorov-Smirnov test

This tool allows to define the importance on the
variable on the basis of the distance between the 'behavioural' distribution and
the 'non-behavioural' one. Behavioural corresponds to the feasibility/viability
area. The opposite applies to the non-behavioural rangwe
[saltelli_global_2008](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184).
This approach is part of the *Monte Carlo Filtering* methodology, more precisely
to the *Regional Sensitivity Analysis* framework. The charts below are shown for
the year 2050. The interested reader is invited to replicate the test for the
other years as exercise.

```{.python .input}
#Smirnov-Kolmorov Test

BVsilverCell=df20503d['silverCell'][df20503d['yearly Delta']>0].sort_values()
BVsilverCellCum = BVsilverCell.cumsum()/BVsilverCell.sum()
NBVsilverCell=df20503d['silverCell'][df20503d['yearly Delta']<0].sort_values()
NBVsilverCellCum = NBVsilverCell.cumsum()/NBVsilverCell.sum()

ax1 = plt.plot(BVsilverCell,BVsilverCellCum,label='silverCell_B',c='b')
plt.plot(NBVsilverCell,NBVsilverCellCum,label='silverCell_NB',c='r')
plt.legend()
```

```{.python .input}
BFsilverCell=df20503d['silverCell'][df20503d['Difference']>0].sort_values()
BFsilverCellCum = BFsilverCell.cumsum()/BFsilverCell.sum()
NBFsilverCell=df20503d['silverCell'][df20503d['Difference']<0].sort_values()
NBFsilverCellCum = NBFsilverCell.cumsum()/NBFsilverCell.sum()

ax1 = plt.plot(BFsilverCell,BFsilverCellCum,label='silverCell_B',c='b')
plt.plot(NBFsilverCell,NBFsilverCellCum,label='silverCell_NB',c='r')
plt.legend()
```

```{.python .input}
BVPVPC=df20503d['PVPC'][df20503d['yearly Delta']>0].sort_values()
BVPVPCCum = BVPVPC.cumsum()/BVPVPC.sum()
NBVPVPC=df20503d['PVPC'][df20503d['yearly Delta']<0].sort_values()
NBVPVPCCum = NBVPVPC.cumsum()/NBVPVPC.sum()

ax1 = plt.plot(BVPVPC,BVPVPCCum,label='PVPC_B',c='b')
plt.plot(NBVPVPC,NBVPVPCCum,label='PVPC_NB',c='r')
plt.legend()
```

```{.python .input}
BFPVPC=df20503d['PVPC'][df20503d['Difference']>0].sort_values()
BFPVPCCum = BFPVPC.cumsum()/BFPVPC.sum()
NBFPVPC=df20503d['PVPC'][df20503d['Difference']<0].sort_values()
NBFPVPCCum = NBFPVPC.cumsum()/NBFPVPC.sum()

ax1 = plt.plot(BFPVPC,BFPVPCCum,label='PVPC_B',c='b')
plt.plot(NBFPVPC,NBFPVPCCum,label='PVPC_NB',c='r')
plt.legend()
```

```{.python .input}
#Smirnov-Kolmorov Test

BVnonPVsilver=df20503d['nonPVsilver'][df20503d['yearly Delta']>0].sort_values()
BVnonPVsilverCum = BVnonPVsilver.cumsum()/BVnonPVsilver.sum()
NBVnonPVsilver=df20503d['nonPVsilver'][df20503d['yearly Delta']<0].sort_values()
NBVnonPVsilverCum = NBVnonPVsilver.cumsum()/NBVnonPVsilver.sum()

ax1 = plt.plot(BVnonPVsilver,BVnonPVsilverCum,label='nonPVsilver_B',c='b')
plt.plot(NBVnonPVsilver,NBVnonPVsilverCum,label='nonPVsilver_NB',c='r')
plt.legend()
```

```{.python .input}
#Smirnov-Kolmorov Test

BFnonPVsilver=df20503d['nonPVsilver'][df20503d['Difference']>0].sort_values()
BFnonPVsilverCum = BFnonPVsilver.cumsum()/BFnonPVsilver.sum()
NBFnonPVsilver=df20503d['nonPVsilver'][df20503d['Difference']<0].sort_values()
NBFnonPVsilverCum = NBFnonPVsilver.cumsum()/NBFnonPVsilver.sum()

ax1 = plt.plot(BFnonPVsilver,BFnonPVsilverCum,label='nonPVsilver_B',c='b')
plt.plot(NBFnonPVsilver,NBFnonPVsilverCum,label='nonPVsilver_NB',c='r')
plt.legend()
```

The two curves represent the cumulative distribution probabilities. The larger
is the maximum distance between the curves, the more important is the input
parameter's uncertainty on the output uncertainty. For an in-depth view of the
methodology, one can have a look at [the above-mentioned
contribution](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184).
The trends for the three variables inquired suggest silver non-PV use may be the
most important variable in 2050 when assessing techno-economic (viability) and
biophysical (feasibility) constraints. Yet the approach is far from being
flawless as it addresses a factor significance, which may be different from
importance (a Smirnov-insignificance result does not necessary imply non-
importance). Furthermore, compensatory effects may occur along with more complex
interactions unable to detect in the test's graphical framework.

### Variance-based Sensitivity Analysis

When one strives for a precise quantitative
information, Sobol sensitivity indices are a very convenient mean to achieve it.
We report here the first-order indices along with total indices in order to
acknowledge for their potential higher order interaction. The model is additive
with the exception of the potential interaction between silver use in PV cells
and power capacity installed. The indices have been computed for the year 2050
and are shown in the table below. One can easily compute the indices also for
other years.

The formula for the estimators are those of [Saltelli et al.
2010](https://www.sciencedirect.com/science/article/pii/S0010465509003087)
Jansen 1999, respectively (references therein, Table 2) for the first-order
sensitivity index and the total sensitivity index. Their expressions are

<!--
Equation labels as ordinary links -->
<div id="_auto2"></div>

$$
\begin{equation}
S_i = V(Y)-\frac{1}{2N}
\sum^N_{j=1}({f(B)_j}-{f(A_B^{(i)})_j})^2
\label{_auto2} \tag{2}
\end{equation}
$$

<!-- Equation labels as ordinary links -->
<div id="_auto3"></div>

$$
\begin{equation}
T_i = \frac{1}{2N} \sum^N_{j=1}({f(A)_j}-{f(A_B^{(i)})_j})^2
\label{_auto3} \tag{3}
\end{equation}
$$

The two sample matrices A and B have
been obtained by slicing column-wise our Sobol sample distributions.

```{.python .input}
# Sensitivity indices plots

df_l = [S_silverCell_df,S_PVPC_df,T_silverCell_df-S_silverCell_df,S_otherAg_df]
df_names = ['S_silverCell_df','S_PVPC_df','S_2order_silverCellPVPC','S_otherAg_df']
colors = ['b','r','g','k']
props = [dict(boxes=co, whiskers=co, medians=co, caps=co) for co in colors]
fig, ax = plt.subplots(figsize=(20,10))
patch = []
for id, dfl in enumerate(df_l):
    dfl.plot(kind='box',ax=ax,label=df_names[id],positions=[c+id/len(df_l) for c in dfl],color=props[id],
            patch_artist=True,showfliers=False)
    plt.ylim(-0.1,1.1)
plt.legend()
plt.show()
```

Black: Silver use in non-photovoltaics; Blue: Silver use in contact
metallization; Red: Expansion of solar power capacity; Green: Second order index
for the interaction between silver use in contact metallization and the
expansion of solar power capacity.

```{.python .input}
fig = plt.figure(figsize=(20,10))
ax = fig.gca(projection='3d')
surf=ax.plot_trisurf(df20503d['silverCell'], df20503d['PVPC'], df20503d['CumulativeSilver'], cmap=plt.cm.jet, linewidth=0.01)
ax.set_xlabel('Silver Contact metallization decrease pace')
ax.set_ylabel('PV Power Capacity increase pace')
ax.set_zlabel('Cumulative Silver 2050 (kton)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
```

```{.python .input}
fig = plt.figure(figsize=(20,10))
ax = fig.gca(projection='3d')
surf=ax.plot_trisurf(df20503d['silverCell'], df20503d['PVPC'], df20503d['yearlySilver'], cmap=plt.cm.jet, linewidth=0.01)
ax.set_xlabel('Silver Contact metallization decrease pace')
ax.set_ylabel('PV Power Capacity increase pace')
ax.set_zlabel('Yearly Silver 2050 (kton)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
```

## Conclusion

The importance of the pace of reducing use of silver for contact
metallization along with the increase in the installed PV power capacity
augments over the course of the years as witnessed by the sensitivity indices.
Higher order interactions between the two parameters can be easily appreciated.