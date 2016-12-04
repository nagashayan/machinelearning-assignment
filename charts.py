import matplotlib.pyplot as plt
import pandas as pd
import pcaTest as pca

plt.figure();


pca_data3 = pd.DataFrame(pca.pca_data3.explained_variance_ratio_)
pca_data3.plot.bar(); plt.axhline(0, color='k')
