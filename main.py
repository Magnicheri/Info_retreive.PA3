import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import accuracy_score as acc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V = 1433
word_col_name = ['w{}'.format(i) for i in range(V)]
data_cites = pd.read_csv("C:\\Users\\pmagn\\OneDrive\\Desktop\\cora\\cora.cites", sep ='\t') #edge
data_contents = pd.read_csv("C:\\Users\\pmagn\\OneDrive\\Desktop\\cora\\cora.content", sep ='\t', names =['paperid'] + word_col_name + ['label'] ) #node

df_nodes =data_contents[word_col_name].to_numpy()

scaler = StandardScaler()
df_nodes_segment = scaler.fit_transform(df_nodes)
pca = PCA(n_components=7, random_state=63)
df_nodes_pca = pca.fit_transform(df_nodes_segment)

adjmat = np.load('Adj_matrix.npy')

df_nodes_adj = np.dot(np.transpose(adjmat), df_nodes_pca)
df_nodes_adj = scaler.fit_transform(df_nodes_adj)

model = KMeans(n_clusters=7, init="k-means++", random_state=220)
model.fit_predict(df_nodes_adj)
print(model.cluster_centers_)


y_hat = model.labels_

y , y_idx = pd.factorize(data_contents['label'])
y_mapping = {y_idx[k]: k for k in range(7)}

transpose = np.transpose(df_nodes_adj)
sum_list =[]

hungarian = []
for k in range(7):
    hungarian.append(np.bincount(y_hat[y==k]))

hg_mapping = linear_sum_assignment(hungarian, maximize= True)

print("Correct mapping lables given with data set: ")
print(hg_mapping[0]),"\n"
print("Labels predicted by model(Hungarain method): ")
print (hg_mapping[1], "\n")

lbl_mapping = [-1]*7
for k in y_mapping:
    masked_labels = model.labels_
    best_cls = np.argmax(np.bincount(model.labels_[y==y_mapping[k]])/np.bincount(model.labels_))
    lbl_mapping[y_mapping[k]] = best_cls

print("Labels predicted by model(LBL method): ")
print(lbl_mapping, "\n")

for m, m_name in ((nmi, 'NMI'), (ari, 'AIR'), (acc, 'ACC')):
       print(f'{m_name:}: {m(y,y_hat):.3f}')

model = KMeans(n_clusters=7, init="k-means++", random_state=220) #250
model.fit_predict(df_nodes_adj)
plt.scatter(df_nodes_adj[:,0], df_nodes_adj[:,1] , c=model.labels_.astype(float))
plt.show()