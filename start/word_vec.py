import numpy as np

import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

"""
    Reference: https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html
"""


def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]


def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.index_to_key), sample)
        else:
            words = list(model.index_to_key)

    word_vectors = np.array([model[w] for w in words])

    n_components = 2
    twodim = PCA(n_components=n_components).fit_transform(word_vectors)

    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.05, y + 0.05, word)
    plt.show()


IN_PATH = 'D:/py_projects/deep_learning/data/'
glove_file = datapath(os.path.join(IN_PATH, 'glove.6B.100d.txt'))
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
print(model.most_similar('obama'))
# [('barack', 0.937216579914093), ('bush', 0.927285373210907), ('clinton', 0.896000325679779), ('mccain', 0.8875633478164673), ('gore', 0.8000321388244629), ('hillary', 0.7933662533760071), ('dole', 0.7851964831352234), ('rodham', 0.7518897652626038), ('romney', 0.7488929629325867), ('kerry', 0.7472624182701111)]

print(model.most_similar('banana'))
# [('coconut', 0.7097253203392029), ('mango', 0.7054824829101562), ('bananas', 0.6887733340263367), ('potato', 0.6629636287689209), ('pineapple', 0.6534532308578491), ('fruit', 0.6519854664802551), ('peanut', 0.6420575976371765), ('pecan', 0.6349173188209534), ('cashew', 0.6294420957565308), ('papaya', 0.6246591210365295)]

print(model.most_similar(negative='banana'))
# [('shunichi', 0.49618104100227356), ('ieronymos', 0.4736502170562744), ('pengrowth', 0.4668096601963043), ('höss', 0.4636845588684082), ('damaskinos', 0.4617849290370941), ('yadin', 0.4617374837398529), ('hundertwasser', 0.4588957726955414), ('ncpa', 0.4577339291572571), ('maccormac', 0.4566109776496887), ('rothfeld', 0.4523947238922119)]

print(model.most_similar(positive=['woman', 'king'], negative=['man']))
# [('queen', 0.7698540687561035), ('monarch', 0.6843381524085999), ('throne', 0.6755736470222473), ('daughter', 0.6594556570053101), ('princess', 0.6520534157752991), ('prince', 0.6517034769058228), ('elizabeth', 0.6464517712593079), ('mother', 0.631171703338623), ('emperor', 0.6106470823287964), ('wife', 0.6098655462265015)]

print(analogy('japan', 'japanese', 'australia'))
# australian

print(analogy('obama', 'clinton', 'reagan'))
# nixon

print(analogy('tall', 'tallest', 'long'))
# longest

print(model.doesnt_match("breakfast cereal dinner lunch".split()))
# cereal

display_pca_scatterplot(model,
                        ['coffee', 'tea', 'beer', 'wine', 'brandy', 'rum', 'champagne', 'water',
                         'spaghetti', 'borscht', 'hamburger', 'pizza', 'falafel', 'sushi', 'meatballs',
                         'dog', 'horse', 'cat', 'monkey', 'parrot', 'koala', 'lizard',
                         'frog', 'toad', 'monkey', 'ape', 'kangaroo', 'wombat', 'wolf',
                         'france', 'germany', 'hungary', 'luxembourg', 'australia', 'fiji', 'china',
                         'homework', 'assignment', 'problem', 'exam', 'test', 'class',
                         'school', 'college', 'university', 'institute'])

display_pca_scatterplot(model, sample=50)
