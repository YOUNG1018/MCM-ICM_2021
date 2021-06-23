import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

N = 10

path = "../data/input_data.csv"

df = pd.read_csv(path, sep=',')

# print(data.corr())

df_corr = df.corr()

data = df_corr.values

fig = plt.figure()

ax = fig.add_subplot(111)

heatmap = ax.pcolor(data,cmap = plt.cm.RdYlGn)

fig.colorbar(heatmap)

ax.set_xticks(np.arange(data.shape[0]),minor = False)

ax.set_yticks(np.arange(data.shape[1]),minor = False)

ax.invert_yaxis()

ax.xaxis.tick_top()

column_labels = df_corr.columns

row_labels = df_corr.index

ax.set_xticklabels(column_labels)

ax.set_yticklabels(row_labels)

plt.xticks(rotation = 90)

heatmap.set_clim(-1,1)

plt.tight_layout()

plt.show()
#
# fig = plt.figure()
#
# sns_plot = sns.heatmap(data, cmap='YlGnBu', xticklabels=False, yticklabels=False, annot=False)
#
# sns_plot.tick_params(labelsize=5, direction='in')
#
# cax = plt.gcf().axes[-1]
# cax.tick_params(labelsize=10, direction='in', top='off', bottom='off', left='off', right='off')
#
# plt.show()
