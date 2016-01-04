#-*-coding:utf-8-*-
from sklearn import datasets
import matplotlib.pyplot as plt
import scipy as sy
import matplotlib.font_manager

fontprop = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
iris = datasets.load_iris()
data = iris.data[:, 2:4]
plt.scatter(data[:,0], data[:,1])
plt.title(u"標準化前", fontdict = {"fontproperties": fontprop})
plt.show()
zdata = sy.stats.zscore(data)
plt.scatter(zdata[:,0], zdata[:,1])
plt.title(u"標準化後", fontdict = {"fontproperties": fontprop})
plt.show()
