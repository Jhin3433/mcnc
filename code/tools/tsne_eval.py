import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
def plot_embedding(X, emb_file=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0) #https://numpy.org/doc/stable/reference/generated/numpy.matrix.min.html
    X = (X - x_min) / (x_max - x_min)


    plt.figure() #https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.figure.html
    # ax = plt.subplot(111)

    plt.scatter(X[:,0], X[:,1], s=2)


    plt.xticks(np.arange(0, 1, step=0.1)), plt.yticks(np.arange(0, 1, step=0.1))
    if emb_file is not None:
        plt.title(emb_file)
    plt.savefig(emb_file + "2"+".png")
def tsne_eval_fun(emb_file=None, embeddings=None):
    if embeddings == None:
        import pickle
        with open(emb_file + ".pickle", "rb") as f:
            embeddings = pickle.load(f)
    
    data = np.array(embeddings.cpu())
    from sklearn.manifold import TSNE
    # euclidean cosine
    transformer = TSNE(n_components=2, metric = "euclidean", init="pca", n_iter=500,n_iter_without_progress=150,n_jobs=2,random_state=0)


    y = np.array([0 for _ in range(data.shape[0])])

    projections = {}
    projections["tsne"] = transformer.fit_transform(data, y)

    plot_embedding(projections["tsne"], emb_file)



def svd_func():
    def transform(embeddings):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=500, n_iter=30, random_state=42)
        svd.fit(np.array(embeddings.cpu()))
        X = [i+1 for i in range(svd.singular_values_.shape[0])]
        print(svd.explained_variance_ratio_.sum())
        Y = svd.singular_values_ / np.linalg.norm(svd.singular_values_)
        return X, Y
    import pickle
    with open("original_test_data_embs.pickle", "rb") as f:
        original_embeddings = pickle.load(f)
    with open("align_test_data_embs.pickle", "rb") as f:
        align_embeddings = pickle.load(f)   

    X_1, Y_1 = transform(original_embeddings)
    X_2, Y_2 = transform(align_embeddings)

    
    plt.plot(X_1, Y_1, X_2, Y_2)
    
    # plt.title(emb_file)
    plt.savefig("svd" + ".png")

if __name__ == "__main__":
    emb_file = "align_test_data_embs" # original_test_data_embs align_test_data_embs
    tsne_eval_fun(emb_file)
    # svd_func()