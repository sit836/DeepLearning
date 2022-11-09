import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD


# corpus = [
#     'I like deep learning.',
#     'I like NLP.',
#     'I enjoy flying.',
# ]

def plot_word_vectors(U, words, method):
    fontsize = 15
    for i in range(len(words)):
        plt.text(U[i, 0], U[i, 1], words[i], fontsize=fontsize)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title(method)
    plt.show()


def plot_singular_vals(s):
    plt.plot(s)
    plt.xlabel('Index')
    plt.ylabel("Singular values")
    plt.show()


words = ["I", "like", "enjoy", "deep", "learning", "NLP", "flying", "."]
cooc_arr = np.array([[0, 2, 1, 0, 0, 0, 0, 0],
                     [2, 0, 0, 1, 0, 1, 0, 0],
                     [1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1],
                     [0, 1, 0, 0, 0, 0, 0, 1],
                     [0, 0, 1, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1, 1, 1, 0],
                     ])

# SVD
U, s, Vh = np.linalg.svd(cooc_arr, full_matrices=False)
print(f'U.shape, s.shape, Vh.shape: {U.shape, s.shape, Vh.shape}')

# For a real symmetric matrix,
#   (1) SVD is equivalent to eigen-decompostion;
#   (2) U and Vh.T are equivalent up to sign changes
print(f'U:\n {np.round(U, 2)}\n')
print(f'Vh.T:\n {np.round(Vh.T, 2)}')
plot_word_vectors(U, words, "SVD")

# TruncatedSVD
plot_singular_vals(s)

n_components = 6
svd_trunc = TruncatedSVD(n_components=n_components, random_state=123)
svd_trunc.fit(cooc_arr)
plot_word_vectors(svd_trunc.components_.T, words, f"TruncatedSVD with n_components={n_components}")
