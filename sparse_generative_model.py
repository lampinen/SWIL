import numpy as np
import matplotlib.pyplot as plot

def _evolve(vector, flip_epsilon, sparsity):
    """For a binary vector, flips 0 valued entries to 1 with probability
    flip_epsilon, preserves with probability 1 - flip_epsilon. For 1-valued 
    entries, it flips to 0 with probability flip_epsilon * (1-sparsity)/sparsity,
    which in expectation preserves the percent 1s at sparsity."""
    # this vector is 1 with probability flip_epsilon, and will be used to
    # replace the zeros
    evolved_zeros = np.random.binomial(1, flip_epsilon, vector.shape)
    # this vector is 1 with probability 
    # (1.-flip_epsilon * (1.-sparsity)/sparsity and will replace the ones
    evolved_ones = np.random.binomial(1, 1.-flip_epsilon * (1.-sparsity)/sparsity, vector.shape)
    # this masks so that it replaces zeros with the evolved zeros, and the ones
    # with the evolved ones
    result = (1.-vector) * evolved_zeros + vector * evolved_ones
    return result 

def sparse_generative_model(n_levels, flip_epsilon, sparsity,
                            num_features=1024, branch_factor=2):
    """Implements a generative model for hierarchically structured data that is
    like that in (Saxe, McClleland, Ganguli, 2018), except that it does
    imbalanced flipping to maintain (in expectation) a desired level of
    sparsity.
    
    Args:
        n_levels: number of levels of branches the hierarchy.
        flip_epsilon: Probability of turning an off bit on at each step.
        sparsity: The desired probability that a bit be on.
        num_features: The feature dimesnionality.
        branch_factor: How many children per node there are at each level of
            the hierarchy.

    outputs:
        data_matrix: binary matrix of size 
            [n_levels**branch_factor, num_features]
    """
    
    current_gen = [np.random.binomial(1, sparsity, [num_features])]
    for lev_i in range(n_levels):
        next_gen = [] 
        for item in current_gen:
            for child_i in range(branch_factor):
                next_gen.append(_evolve(item, flip_epsilon, sparsity))
        current_gen = next_gen 
    return np.array(current_gen)


if __name__=="__main__":
    # some tests
    num_evolve_test = 100
    np.random.seed(0)

    vec = np.array([1] * 20 + [0] * 80)
    res = np.zeros([num_evolve_test, 100])
    for i in range(num_evolve_test):
        res[i, :] = _evolve(vec, 0.05, 0.2)

    print(np.mean(res[:, :20]), np.mean(res[:, 20:]))
    print(np.mean(res[:, :]))

    vec = np.array([1] * 20 + [0] * 80)
    res = np.zeros([num_evolve_test, 100])
    for i in range(num_evolve_test):
        vec = _evolve(vec, 0.05, 0.2)
        res[i, :] = vec 

    print(np.mean(res[:, :20]), np.mean(res[:, 20:]))
    print(np.mean(res[:, :]))

    X = sparse_generative_model(7, 0.005, 0.1)
    plot.figure()
    plot.imshow(np.matmul(X, X.transpose()))
    plot.savefig("sparse_hierarchy/correlation.png")

    #print(np.matmul(X, X.transpose()))
    U, S, V = np.linalg.svd(X, full_matrices=False)

    plot.figure()
    plot.imshow(U)
    plot.savefig("sparse_hierarchy/U.png")

    plot.figure()
    plot.plot(range(len(S)), S)
    plot.savefig("sparse_hierarchy/S.png")


    plot.figure()
    plot.imshow(V)
    plot.savefig("sparse_hierarchy/V.png")

