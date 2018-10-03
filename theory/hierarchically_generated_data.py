import numpy as np

def hierarchically_generated_data(n_splits, n_children, f_per_child):
    """Creates a hierarchically strutured dataset from a tree of depth n_splits
    with n_children children per node, and f_per_child binary indicator
    features for each child."""
    num_items = n_children**(n_splits-1)
    num_features = [f_per_child*n_children**i for i in range(n_splits)]
    part_sum_features = [sum(num_features[:i]) for i in range(n_splits)]
    data = np.zeros([num_items, sum(num_features)]) 
    for i, nf in enumerate(num_features):
        start = part_sum_features[i]
        num_levels = n_children**i
        items_per_level = num_items/num_levels 
        for l in range(num_levels):
            data[l*items_per_level:(l+1)*items_per_level, start+f_per_child*l:start+f_per_child*(l+1)] = 1. 

    return data
        

if __name__ == "__main__":
    print(hierarchically_generated_data(2, 3, 1))
    print(hierarchically_generated_data(3, 2, 1))
    print(hierarchically_generated_data(2, 3, 2))
    print(hierarchically_generated_data(3, 2, 2))
