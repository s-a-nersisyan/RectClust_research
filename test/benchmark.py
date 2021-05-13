import pandas as pd
import numpy as np

from RSD_mixture import RSDMixtureEM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from ExKMC.Tree import Tree

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ari_ci import ari_ci

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

np.random.seed(17)

# Format: {dataset: [target_col_number, [col_numbers_to_drop]]}
datasets = {
    "TCGA-BRCA": [4, []],
    "iris": [4, []],
    "wine": [0, []],
    "wdbc": [1, [0]],
    "optdigits": [64, []],
    "mice_protein": [77, []],
    "anuran": [22, []],
}

# Format: {name: [class, n_clusters_arg_name, init_kwargs]}
models = {
    "RSD mixture": [RSDMixtureEM, "n_clusters", {"n_init": 100}],
    "Gaussian mixture diag": [GaussianMixture, "n_components", {"covariance_type": "diag", "n_init": 100}],
    "Gaussian mixture full": [GaussianMixture, "n_components", {"covariance_type": "full", "n_init": 100}],
    "KMeans": [KMeans, "n_clusters", {"n_init": 100}],
    "ExKMC": [Tree, "k", {}]
}

res = pd.DataFrame(columns=[
    "Dataset", "N samples", "N features", "N classes", "Model", "ARI", "ARI CI low", "ARI CI high"
])

for dataset in datasets:
    df = pd.read_csv("datasets/{}.data".format(dataset), header=None)
    X = df.drop(columns=[datasets[dataset][0]] + datasets[dataset][1]).to_numpy()
    y = df[datasets[dataset][0]].to_numpy()
    
    X = SimpleImputer().fit_transform(X)
    X = StandardScaler().fit_transform(X)

    for model_name, model in models.items():
        kwargs = {model[1]: len(np.unique(y)), **model[2]}
        model = model[0](**kwargs)
        
        model.fit(X)
        y_pred = model.predict(X)
        
        ARI, _, ARI_low, ARI_high = ari_ci(y_pred, y, alpha=0.05)
        res.loc[len(res)] = [dataset, X.shape[0], X.shape[1], len(np.unique(y)), model_name, ARI, ARI_low, ARI_high]

res.to_csv("bench.tsv", sep="\t")
