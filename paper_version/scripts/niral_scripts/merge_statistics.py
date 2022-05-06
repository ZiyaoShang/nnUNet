import numpy as np


def mergestats(young, old, joint, save):
    y = np.load(young)
    o = np.load(old)
    j = np.load(joint)
    assert y.shape == o.shape == j.shape
    print(y.shape)
    final = np.copy(j)

    inds = np.array([4, 5, 6, 13, 25, 26, 30, 42, 43])
    assert np.sum(y[:, inds]) == 0
    final[:, inds] = o[:, inds]

    assert np.sum(o[:, 8]) == 0
    final[:, 8] = y[:, 8]

    print("y: ")
    print(y)
    print("o: ")
    print(o)
    print("j: ")
    print(j)
    print("f: ")
    print(final)

    np.save(save, final)

    saved = np.load(save)
    assert np.sum(final) == np.sum(saved)

mergestats(young="/home/ziyaos/Desktop/look/BEE_plus/all_choices/T2_prior/young/prior_means.npy",
           old="/home/ziyaos/Desktop/look/BEE_plus/all_choices/T2_prior/old/prior_means.npy",
           joint="/home/ziyaos/Desktop/look/BEE_plus/all_choices/T2_prior/joint/prior_means.npy",
           save="/home/ziyaos/Desktop/look/BEE_plus/all_choices/T2_prior/merged/prior_means.npy")
