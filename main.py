import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import accuracy_score as acc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

V = 1433
word_col_name = ['w{}'.format(i) for i in range(V)]
data_cites = pd.read_csv("C:\\Users\\magni\\Desktop\\cora\\cora.cites", sep ='\t') #edge
data_contents = pd.read_csv("C:\\Users\\magni\\Desktop\\cora\\cora.content", sep ='\t', names =['paperid'] + word_col_name + ['label'] ) #node

df_nodes =data_contents[word_col_name].to_numpy()

scaler = StandardScaler()
df_nodes_segment = scaler.fit_transform(df_nodes)
pca = PCA(n_components= 15, svd_solver='arpack')#7 #15
df_nodes_pca = pca.fit_transform(df_nodes_segment)

adjmat = np.load('Adj_matrix.npy')

df_nodes_adj = np.dot(np.transpose(adjmat), df_nodes_pca)

model = KMeans(n_clusters=7, init="k-means++", n_init =50, max_iter=500, random_state= 613)#4693 #30230 #613(50,1000), 581
model.fit_predict(df_nodes_adj)

y_hat = model.labels_

y , y_idx = pd.factorize(data_contents['label'])
y_mapping = {y_idx[k]: k for k in range(7)}

transpose = np.transpose(df_nodes_adj )
sum_list =[]

hungarian = []
for k in range(7):
    hungarian.append(np.bincount(y_hat[y == k]))


_, hg_mapping = linear_sum_assignment(hungarian, maximize=True)
acc_temp = acc([hg_mapping[i] for i in y], y_hat)

lbl_mapping = [-1]*7
for k in y_mapping:
    masked_labels = model.labels_
    best_cls = np.argmax(np.bincount(model.labels_[y==y_mapping[k]])/np.bincount(model.labels_))
    lbl_mapping[y_mapping[k]] = best_cls
print("LBL mapping:")
print(lbl_mapping, "\n")
print("Hungarin mapping:")
print(hg_mapping,"\n")
print("Hungarian Accuracy:")
print(acc_temp, "\n")

print("NMI and ARI:")
for m, m_name in ((nmi, 'NMI'), (ari, 'AIR')):
       print(f'{m_name:}: {m(y,y_hat):.3f}')



tsne = TSNE(n_components=3, learning_rate='auto', n_iter=500, init= 'random')
df_nodes_tsne = tsne.fit_transform(df_nodes_adj)


df_names = (data_contents['paperid'])

plt.scatter( df_names, y_hat, c=model.labels_.astype(float))
plt.show()

plt.scatter(df_names, y , c=model.labels_.astype(float))
plt.show()



centroids = model.cluster_centers_
u_label = np.unique(y_hat)


for i in u_label:
    plt.scatter(df_nodes_adj[y_hat == i , 0], df_nodes_adj[y_hat == i, 1], label = i, s =5)
plt.scatter(centroids[:,0], centroids[:,1], s = 20, color = 'k')
plt.legend()
plt.show()

for i in u_label:
    plt.scatter(df_nodes_tsne[y_hat == i , 0], df_nodes_tsne[y_hat == i, 1], label = i, s =10)
plt.legend()
plt.show()
