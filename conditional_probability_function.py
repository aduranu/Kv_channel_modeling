import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import math


# # P(open|[K+]) Conditional probability p(y|x) (from stat. mech. and Sigworth's model)

# # Assumptions
# The cell is within the range of standard physiological conditions
# Only changes in [K+ out] are significant for the cell's sensing.
# The channel state can be evaluated for arbitrary intervals of x
# The cell's permeabilities to K+, Na+, and Cl- ions are the standard values for a neuron at rest
# (which determine the potential "perceived")

# # Arguments
# x: To output only one prob for a specific concentration (in mM) value in the environment.
# x_max: To output whole prob distribution for a "range" of the environment - maximum K+ extracellular concentration
# as [K+ out] in mM units.
# standard_ions: if True, calculates ion concentration ratios with average standard values if False, the intracellular
# and extracellular mM concentrations have to be specified as na_in, na_out, k_in, cl_in, cl_out
# temp: temperature of environment surrounding the cell
# q: gating current (in terms of elemental charges e0).
# G_no_potential: Free energy change from closed to open state at zero membrane potential
# x_interval: interval that separates the environment values to be evaluated, not necessary if x=True

# For q and G_no_potential: To see overlaying plots with different values of each, write values as a list.
# Only one can be a list at a time, with the other argument being an int. If no overlaying plots are desired, write any
# as a list of one element and the other as int.

# # Returns
# value: if x is not None, returns one probability value for the given x value
# graph: if x_max is not None and plot is True, returns P(open|[K+]) function as a graph
# DataFrame: if x_max is not None and plot is False, returns pandas df with x values as rows and 1, 0 as columns
# (1 open, 0 closed)


def p_y_x(q=None, G_no_potential=None, temp=298, x_max=None, x=None, x_interval=1.0, plot=False, standard_ions=True,
          na_in=0, na_out=0, k_in=0, cl_in=0, cl_out=0):
    R = 8.3144
    F = 96485
    Kb = 1.3806
    if standard_ions:
        condition = (140 + 0.05 * 10 + 0.45 * 110)
        condition2 = (0.05 * 145 + 0.45 * 4)
    else:
        condition = (k_in + 0.05 * na_in + 0.45 * cl_out)
        condition2 = (0.05 * na_out + 0.45 * cl_in)

    if x_max:
        ax = plt.subplot(111)
        if type(q) is list:
            for k in q:
                x = list(np.arange(0, x_max, x_interval))
                probs = []
                for interval in x:
                    potential = ((R * temp * 1000 / F) * np.log((interval / condition) + (condition2 / condition)))
                    prob = [(G_no_potential / (Kb * temp)) - ((k * potential) / (Kb * temp))]
                    exp = np.exp(prob)
                    probs.append(1 / (1 + exp[0]))
                df = pd.DataFrame(probs, index=x, columns=['P(open|c)'])
                if not plot:
                    df['P(closed|c)'] = [1 - i for i in probs]
                    df.columns = ['open', 'closed']
                    return df
                if plot:
                    df.plot(ax=ax)
            ax.legend([str(k) for k in q])
            ax.set_ylabel('P (open)')
            ax.set_xlabel('[K+] (mM)')
            return plt.show()
        if type(G_no_potential) is list:
            for k in G_no_potential:
                x = list(np.arange(0, x_max, x_interval))
                probs = []
                for interval in x:
                    potential = ((R * temp * 1000 / F) * np.log((interval / condition) + (condition2 / condition)))
                    prob = [(k / (Kb * temp)) - ((q * potential) / (Kb * temp))]
                    exp = np.exp(prob)
                    probs.append(1 / (1 + exp[0]))
                df = pd.DataFrame(probs, index=x, columns=['P(open|c)'])
                if not plot:
                    df['P(closed|c)'] = [1 - i for i in probs]
                    df.columns = ['open', 'closed']
                    return df
                if plot:
                    df.plot(ax=ax)
            ax.legend([str(k) for k in G_no_potential])
            ax.set_ylabel('P (open)')
            ax.set_xlabel('[K+] (mM)')
            return plt.show()

    if x:
        potential = ((R * temp * 1000 / F) * np.log((x / condition) + (condition2 / condition)))
        prob = [(G_no_potential / (Kb * temp)) - ((q[0] * potential) / (Kb * temp))]
        exp = np.exp(prob)
        return 1 / (1 + exp[0])