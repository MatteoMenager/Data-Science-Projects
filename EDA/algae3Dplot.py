import numpy as np
import pandas as pd

# statistical modules
import scipy.stats as stats 
import statsmodels.api as sm 
import pingouin as pg

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets

# set the default style for plots
plt.style.use("seaborn-darkgrid")

# import data from cat_pcs.pkl file
df = pd.read_pickle('data/04-Algae_Blooms/src/cat_pcs.pkl')

# 3D scatter plot of PC1 PC2 PC3 with hue season
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['PC1'], df['PC2'], df['PC3'], c=pd.factorize(df['Size'])[0])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

                  