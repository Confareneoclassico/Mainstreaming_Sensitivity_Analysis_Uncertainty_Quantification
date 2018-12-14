# NUSAP visualization tool

<!-- AUTHOR: Samuele Lo Piano -->
<!-- AUTHOR:
Jeroen Van der Sluijs -->
**Samuele Lo Piano**, **Jeroen Van der Sluijs** <br/>
[slopiano@gmail.com](mailto:s.lopiano@gmail.com)

```{.python .input}
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
```

```{.python .input}
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import seaborn as sns
import numpy as np
```

```{.python .input}
def f(x):
    return x
```

Please select whether you have an excel spreadsheet already available or you
would rather interactively compile the NUSAP scores. If you have a file
available, the first column should include the experts names (e.g. cell A2:
'Expert1', cell A3: 'Expert 2' and so forth). The other columns should be about
the criteria addressed (e.g. cell B1: 'proxy', cell C1: 'validation', cell D1:
'reliability' and so forth). Please leave cell A1 empty. The cells are to be
filled with the score a given expert attributes to a given proxy: e.g. cell 'B2'
shall contain Expert 1's score on 'proxy'. Scores shall be included in the range
0-4.

```{.python .input}
chart = interactive(f, x=['file','compile'])
display(chart)
```

**You can skip the following cell if you do not have a file available.
Otherwise, please insert your file path, including the document name.**
**Extensions compatible: .xls, .xlsx**

```{.python .input}
if chart.children[0].value=='file':
  path = widgets.Text()
  display(path)
```

You can skip to the **plot** cell if you already have a file available. <br/>
Otherwise, select the number of experts.

```{.python .input}
Expert = widgets.Dropdown(options=[e for e in range(2,9)],value=2,description='Experts:',disabled=False)
display(Expert)
```

Select the number of criteria and their names.

```{.python .input}
Criteria = widgets.Dropdown(options=[c for c in range(2,11)],value=3,description='Criteria:',disabled=False,)
display(Criteria)
```

```{.python .input}
accordion = widgets.Accordion(children=[widgets.Text() for cr in range(Criteria.value)])
for cr in range(Criteria.value):
    accordion.set_title(cr, 'Criterion'+str(cr+1))
accordion
```

Attribute a score to those criteria.

```{.python .input}
tab_content = [['value' for cr in range(Criteria.value)] for ex in range(Expert.value)]
children = [[widgets.Text(description=name) for name in tab_content[ex]] for ex in range(Expert.value)]
tab = [widgets.Tab() for ex in range(Expert.value)]
for ex in range(Expert.value):
    tab[ex].children = children[ex]
    keys = []
    for cr in range(Criteria.value):
        tab[ex].set_title(cr, accordion.children[cr].value)
        keys.append(accordion.children[cr].value)
    display(tab[ex])
```

**Continue from here if you have selected the file option**. Set the dataframe for the final plot.

```{.python .input}
# Set data
if chart.children[0].value=='compile':
  Se = pd.Series(["Expert "+str(ex+1) for ex in range(Expert.value)])  
  df = pd.DataFrame({k:[tab[ex].children[index].value for ex in range(Expert.value)] for index, k in enumerate(keys)}, \
    index=[Se.values[ex] for ex in range(Expert.value)])
if chart.children[0].value=='file':
  df = pd.read_excel(path.value)
```

Choose whether you would like to visualise the final data in a radar or boxplot.

```{.python .input}
plot = interactive(f, x=['radar','boxplot'])
display(plot)
```

It is time to finally visualise your plot. :-)

```{.python .input}
#Radar plot based on https://python-graph-gallery.com/ - check it out, it is full of amazing features! :-)
if plot.children[0].value == 'radar':
        # ------- PART 1: Create background

    # number of variable
    categories=list(df)
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1,2,3], ["1","2","3"], color="grey", size=7)
    plt.ylim(0,4)

    colors = ['b','g','r','c','m','y','k','w']

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data

    for ex in range(len(df)):
        values=df.iloc[ex].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, colors[ex], linewidth=1, linestyle='solid', label="expert "+str(ex+1))
        ax.fill(angles, values, colors[ex], alpha=0.1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()
else:
    plt.imshow(np.linspace(0, 1, 100).reshape(1, -1), extent=[-0.25, 4.25, -1, len(df.columns)], aspect='auto', cmap='RdYlGn_r')
    ax = sns.boxplot(data=df, orient="h", color='#d8dcd6')
```
