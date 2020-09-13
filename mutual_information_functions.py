import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import math

# # I(x;y) Mutual information

# # Assumptions
# The events (possible states of the channel) are discrete
# The possible values of the environment (x) follow a specific probability distribution p(x)

# # Arguments
# x_max: maximum K+ extracellular concentration as [K+ out] in mM units.
# temp: temperature of environment surrounding the cell (Kelvin)
# q: gating current (in terms of elemental charges e0). Integer.
# G_no_potential: Free energy change from closed to open state at zero membrane potential. Integer.
# x_interval: interval that separates the environment values to be evaluated, not necessary if x=True
# verbose: if True, print the value of gating current corresponding to the output

# **other_px: other parameters for different px distributions
# sigma: standard deviation used in normal and bimodal distributions
# lam: lambda value used in poisson distribution

# # Returns
# value: mutual information of the system, in bits

from conditional_probability_functions import p_y_x


def I_xy(x_max=20, G_no_potential=0, q=10, temp=298, x_interval=1.0, verbose=False, mean=1, mean2=1, exp='',
         **other_px):
    a = list(np.arange(0, x_max, x_interval))
    get_py = []
    probs = np.random.dirichlet(np.ones(len(a)), size=1)
    px_sum = []

    for i in a[1:]:
        if exp == 'exponential':
            px = (1 / mean) * np.exp(-(1 / mean * i))
        if exp == 'bimodal':
            px = (1 / (other_px['sigma'] * np.sqrt(2 * np.pi))) * np.exp(
                (-((i - mean) ** 2)) / (2 * (other_px['sigma'] ** 2))) + (
                         1 / (other_px['sigma'] * np.sqrt(2 * np.pi))) * np.exp(
                (-((i - mean2) ** 2)) / (2 * (other_px['sigma'] ** 2)))
        if exp == 'normal':
            px = (1 / (other_px['sigma'] * np.sqrt(2 * np.pi))) * np.exp(
                (-((i - mean) ** 2)) / (2 * (other_px['sigma'] ** 2)))
        if exp == 'uniform':
            px = 1 / len(a)
        if exp == 'random':
            px = np.random.choice(probs[0], 1)[0]
        if exp == 'poisson':
            px = ((other_px['lam'] ** i) * (np.exp(-(other_px['lam'])))) / (math.gamma(i + 1.0))
        px_sum.append(px)
    px_norm = sum(px_sum)

    n = -1
    for i in a[1:]:
        n += 1
        pyx = p_y_x(q=[q], G_no_potential=G_no_potential, temp=temp, x=i)
        px = px_sum[n]
        get_py.append((px / px_norm) * pyx)
    py = sum(get_py)

    I_val = []
    n = -1
    for i in a[1:]:
        n += 1
        pyx = p_y_x(q=[q], G_no_potential=G_no_potential, temp=temp, x=i)
        px = px_sum[n]
        to_sum = (px / px_norm) * pyx * np.log2(pyx / py)
        I_val.append(to_sum)

    if verbose:
        print('With gating current: ', q, '  Mutual information is:')
    return sum(I_val)


# # Visualize the dependence of mutual information on q

# # Arguments
# q_range: maximum value to be considered for gating current
# x_max: maximum K+ extracellular concentration as [K+ out] in mM units.
# G_no_potential: Free energy change from closed to open state at zero membrane potential. Integer.
# temp: temperature of environment surrounding the cell (Kelvin)
# exp: list with names (str) of distributions to be considered. If only one is desired, write a list with 1 element
# x_interval: interval that separates the environment values to be evaluated

# # Returns
# graph: plot of the mutual information vs gating current. Overlaying plots if exp has more than 1 element

def I_xy_q(q_range=20, x_max=20, G_no_potential=0, temp=298, exp=['normal'], x_interval=1.0, **other_px):
    ax = plt.subplot(111)
    for e in exp:
        y_val = []
        for i in list(range(1, q_range)):
            val = I_xy(x_max=x_max, q=i, x_interval=x_interval, temp=temp, G_no_potential=G_no_potential, exp=e,
                       **other_px)
            y_val.append(val)
        df = pd.DataFrame(y_val, index=list(range(1, q_range)), columns=['Mutual information'])
        df.plot(ax=ax, legend=False)
    ax.legend(exp)
    ax.set_ylabel('I(x;y) (bits)')
    ax.set_xlabel('Gating current (e°)')
    return plt.show()


# # Visualize the dependence of mutual information on G_no_potential

# # Arguments
# q: gating current. Integer.
# x_max: maximum K+ extracellular concentration as [K+ out] in mM units.
# G_range: range of values to be considered for free energy at zero potential as a list: [lowest end, highest end]
# temp: temperature of environment surrounding the cell (Kelvin)
# exp: list with names (str) of distributions to be considered. If only one is desired, write a list with 1 element
# x_interval: interval that separates the environment values to be evaluated

# # Returns
# graph: plot of the mutual information vs free energy. Overlaying plots if exp has more than 1 element

def I_xy_G(q=20, x_max=20, G_range=[0, 1], temp=298, exp='exp', x_interval=1.0, **other_px):
    y_val = []
    for i in list(range(G_range[0], G_range[1])):
        val = I_xy(x_max=x_max, q=q, temp=temp, G_no_potential=i, exp=exp, x_interval=x_interval, **other_px)
        y_val.append(val)
    df = pd.DataFrame(y_val, index=list(range(G_range[0], G_range[1])), columns=['Mutual information'])
    df.plot()
    plt.ylabel('I(x;y) (bits)')
    plt.xlabel('G° free energy')
    return plt.show()


# # Visualize the evolution of mutual information in Kv channel types

# # Arguments
# values: list, where each item is a tuple with ('name', q, G_no_potential) for each Kv channel case, where:
# name: (string) type of Kv channel
# q: gating current (in terms of elemental charges e0)
# G_no_potential: Free energy change from closed to open state at zero membrane potential (ideally negative)
# Introduce "values" in evolutionary order: from basal to most recently evolved

# x_max: maximum K+ extracellular concentration as [K+ out] in mM units.
# x_interval: interval that separates the environment values to be evaluated, not necessary if x=True
# temp: temperature of environment surrounding the cell (Kelvin)
# exp: list with names (str) of distributions to be considered. If only one is desired, write a list with 1 element
# compare_q: if True, plots mutual information values and the gating current for each Kv channel.

# # Returns
# graph: mutual information values of the Kv channels in "values", as organized in the arguments. Overlaying plots if
# exp has more than 1 element

def I_evolution(values=[], x_max=160, x_interval=1.0, temp=298, exp=['normal'], compare_q=False, **other_px):
    ax = plt.subplot(111)
    pxs = []
    for e in exp:
        pxs.append(e)
        names = []
        v = []
        for i in values:
            names.append(i[0])
            val = I_xy(x_max=x_max, G_no_potential=i[2], q=i[1], temp=temp, x_interval=x_interval, exp=e, **other_px)
            v.append(val)
        pl = pd.DataFrame(v, index=names, columns=['Mutual information'])
        pl.plot(ax=ax, legend=False)
        ax.legend(pxs)

    if compare_q:
        q_values = []
        for i in values:
            q_values.append(i[1])
        ax2 = ax.twinx()
        a = pd.DataFrame(q_values, index=names)
        a.plot(ax=ax2, legend=False, color='red', linestyle='dashed')
        ax2.set_ylabel('Gating current (e°)')
        ax.set_xlabel('Kv channels in evolutionary order')
        ax.set_ylabel('I(X;Y) (bits)')
        ax.yaxis.label.set_color('blue')
        ax2.yaxis.label.set_color('red')
        return plt.show()

    if not compare_q:
        ax.set_xlabel('Kv channels in evolutionary order')
        ax.set_ylabel('I(X;Y) (bits)')
        return plt.show()
