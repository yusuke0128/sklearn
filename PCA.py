#-*-coding:utf-8-*-
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import scipy as sy

print("入力画像のファイル名を入力してください\n")
orgName = raw_input()
orgArray = np.asarray(Image.open(orgName).convert('L'))
print("主成分分析後の次元数を入力してください\n")
n = input()
pca = PCA(n_components=n)
pca.fit(orgArray)
pca_res = pca.transform(orgArray)
img = pca.inverse_transform(pca_res)
print("出力画像のファイル名を入力してください\n")
name = raw_input() 
sy.misc.imsave(name+'.jpg', img)
