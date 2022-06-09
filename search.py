from audioop import cross
import numpy as np
import time
import sys

# Euclidean Distance Function
def euclidean_distance(p, q):
    dist = 0
    for i in range(len(p)):
        dist += (p[i] - q[i]) ** 2
    dist = dist ** 0.5
    return dist

# Create subset of data by adding a feature
def create_data_subset_fw(x, current_features, j):
    x_sub = []
    c_f = current_features.copy()
    c_f.append(j)
    for i in range(len(x)):
        row = []
        for j in range(len(x[i])):
            if j in c_f:
                row.append(x[i][j])
        x_sub.append(row)
    return x_sub

# Create subset of data by removing a feature
def create_data_subset_bw(x, current_features, j):
    x_sub = []
    c_f = current_features.copy()
    c_f.remove(j)
    for i in range(len(x)):
        row = []
        for j in range(len(x[i])):
            if j in c_f:
                row.append(x[i][j])
        x_sub.append(row)
    return x_sub

# Leave-one-out cross validation function
def cross_validation_using_NN(x, y):
    k = len(x)

    # Variables to calculate accuracy
    correct_classifications = 0
    total_classificaitons = 0

    # Find nearest neighbor for each data point
    for i in range(k):
        p = x[i]
        c = y[i]
        min_dist = float("inf")
        min_class = None
        for j in range(k):
            if i != j:
                if euclidean_distance(p, x[j]) < min_dist:
                    min_dist = euclidean_distance(p, x[j])
                    min_class = y[j]
        if (min_class == c):
            correct_classifications += 1
        total_classificaitons += 1
    
    # Calculate accuracy
    acc = correct_classifications / total_classificaitons
    return acc

# Search using forward selection
def forward_selection(x, y):
    n_features = len(x[0])
    current_features = []

    # Variables to find overall best feature set
    features_selected = []
    best_acc = 0

    # Timer start
    start_time = time.time()

    # Variables to save plotting points
    accs = []
    feats = []

    # Add default rate
    accs.append(cross_validation_using_NN(create_data_subset_bw(x, [0], 0), y))
    feats.append(-1)

    # Loop for levels
    for i in range(n_features):
        print("On search tree level: ", i+1)

        # Variables to find best feature set at current tree level
        feature_to_add_at_this_level = None
        best_so_far_acc = 0

        for j in range(n_features):
            if j not in current_features:
                x_sub = create_data_subset_fw(x, current_features, j)
                acc = cross_validation_using_NN(x_sub, y)
                print("\tConsidering adding feature", j+1, "with accuracy:", acc)
                if acc > best_so_far_acc:
                    best_so_far_acc = acc
                    feature_to_add_at_this_level = j

        current_features.append(feature_to_add_at_this_level)
        print("\tAdded feature", feature_to_add_at_this_level+1)
        accs.append(best_so_far_acc)
        feats.append(feature_to_add_at_this_level+1)

        if best_so_far_acc > best_acc:
            features_selected = current_features.copy()
            best_acc = best_so_far_acc

        # Break loop after 8 hours
        elapsed = time.time() - start_time
        if elapsed > (8*60*60):
            break

    for i in range(len(features_selected)):
        features_selected[i] += 1

    print("Best features:", features_selected)
    print("Time taken: ", elapsed)

    # Save plot variables
    plot_val = np.array([feats,accs])
    # np.savetxt('plot_val_fw.txt', plot_val, delimiter=',')
        
    return features_selected

def backward_elimination(x, y):
    n_features = len(x[0])
    current_features = list(range(n_features))

    # Variables to calculate overall best feature set
    features_selected = list(range(n_features))
    best_acc = cross_validation_using_NN(x, y)

    # Timer start
    start_time = time.time()

    # Variables to save plotting points
    accs = []
    feats = []

    # Add accuracy for all features
    accs.append(cross_validation_using_NN(x, y))
    feats.append(-1)

    # Loop for levels
    for i in range(n_features):
        print("On search tree level: ", i+1)

        # Variables to find best feature set at current tree level
        feature_to_rm_at_this_level = None
        best_so_far_acc = 0

        for j in range(n_features):
            if j in current_features:
                x_sub = create_data_subset_bw(x, current_features, j)
                acc = cross_validation_using_NN(x_sub, y)
                print("\tConsidering removing feature",j+1, "with accuracy:", acc)
                if acc > best_so_far_acc:
                    best_so_far_acc = acc
                    feature_to_rm_at_this_level = j
        
        current_features.remove(feature_to_rm_at_this_level)
        print("\tRemoved feature", feature_to_rm_at_this_level+1)
        accs.append(best_so_far_acc)
        feats.append(feature_to_rm_at_this_level+1)

        if best_so_far_acc > best_acc:
            features_selected = current_features.copy()
            best_acc = best_so_far_acc
        
        # Break loop after 8 hours
        elapsed = time.time() - start_time
        if elapsed > (8*60*60):
            break

    for i in range(len(features_selected)):
        features_selected[i] += 1

    print("Best features:", features_selected)
    print("Time taken: ", elapsed)

    # Save plot variables
    plot_val = np.array([feats,accs])
    # np.savetxt('plot_val_bw.txt', plot_val, delimiter=',')
    
    return features_selected

def driver():
    dataset = int(input("Please choose a dataset\n1: Small\n2: Large\nYour choice: "))
    if dataset == 1:
        data = np.loadtxt('small.txt')
    elif dataset != 2:
        data = np.loadtxt('large.txt')
    else:
        sys.exit("Please choose a valid option.")
    
    search = int(input("Please choose a search function\n1: Forward Selection\n2: Backward Elimination\nYour choice: "))
    if search != 1 and search != 2:
        sys.exit("Please choose a valid option.")

    # Extract features and classes
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i][1:])
        y.append(int(data[i][0]))

    if search == 1:
        features = forward_selection(x, y)
    else:
        features = backward_elimination(x, y)

    return features

if __name__ == "__main__":
    driver()