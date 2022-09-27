import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st
import scipy.cluster.hierarchy as sch
import sklearn
from sklearn.cluster import AgglomerativeClustering 

if __name__ == '__main__':
    # parameters = np.loadtxt("./results/lmnist-post/glifr_lheta/trial_0/lightning_logs/version_4754407/learnedparams.csv", delimiter=',')
    parameters = np.loadtxt("./results/nmnist-post/glifr_lheta/trial_0/lightning_logs/version_0/learnedparams.csv", delimiter=',')

    n_clusters_list = np.arange(2, 12, step=1)
    ch_scores = []

    for n_clusters in n_clusters_list:
        hc = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'euclidean', linkage ='ward')
        y_hc=hc.fit_predict(parameters)
        ch_scores.append(sklearn.metrics.calinski_harabasz_score(parameters, y_hc))
    results = np.zeros((len(ch_scores), 2))
    results[:, 0] = n_clusters_list.reshape(-1)
    results[:, 1] = ch_scores
    # np.savetxt("./results/lmnist-post/glifr_lheta/trial_0/lightning_logs/version_4754407/chscores.csv", results, delimiter=',')
    np.savetxt("./results/nmnist-post/glifr_lheta/trial_0/lightning_logs/version_0/chscores.csv", results, delimiter=',')

    best_n_clusters = n_clusters_list[np.argmax(ch_scores)]
    if best_n_clusters == 2:
        best_n_clusters = 1
    print(f"using n_clusters {best_n_clusters}")

    hc = AgglomerativeClustering(n_clusters = best_n_clusters, affinity = 'euclidean', linkage ='ward')
    y_hc=hc.fit_predict(parameters)
    # np.savetxt("./results/lmnist-post/glifr_lheta/trial_0/lightning_logs/version_4754407/cluster_labels.csv", y_hc, delimiter=',')
    np.savetxt("./results/nmnist-post/glifr_lheta/trial_0/lightning_logs/version_0/cluster_labels.csv", y_hc, delimiter=',')