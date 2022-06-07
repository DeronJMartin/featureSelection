import numpy as np
import time

def euclidean_distance(p, q):
    dist = 0
    for i in range(len(p)):
        dist += (p[i] - q[i]) ** 2
    dist = dist ** 0.5
    return dist

# def neareast_neighbor_classifier(X_train, y_train, p):
#     y_pred = []
#     min_dist = float("inf")
#     min_class = None
#     for i in range(len(X_train)):
#         if euclidean_distance(p, X_train[i]) < min_dist:
#             min_dist = euclidean_distance(p, X_train[i])
#             min_class = y_train[i]
#     y_pred.append(min_class)
#     return y_pred

def create_data_subset(x, current_features, j):
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

def cross_validation_using_NN(x, y):
    # Leave-one-out cross validation
    k = len(x)

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

def forward_selection(x, y):
    features_selected = []
    n_features = len(x[0])

    # Base case: 0 features, randomly guess class
    current_features = []
    best_acc = 1/(len(set(y)))

    # Timer start
    start_time = time.time()

    # Loop for levels
    for i in range(n_features):
        print("On search tree level: ", i+1)

        feature_to_add_at_this_level = None
        best_so_far_acc = 0

        for j in range(n_features):
            if j not in current_features:
                x_sub = create_data_subset(x, current_features, j)
                acc = cross_validation_using_NN(x_sub, y)
                print("\tConsidering adding feature ", j+1, "with accuracy: ", acc)

                if acc > best_so_far_acc:
                    best_so_far_acc = acc
                    feature_to_add_at_this_level = j

        current_features.append(feature_to_add_at_this_level)
        print("\tAdded feature", feature_to_add_at_this_level+1)

        if best_so_far_acc > best_acc:
            features_selected = current_features.copy()
            best_acc = best_so_far_acc

        elapsed = time.time() - start_time

        # Break loop after 8 hours
        if elapsed > (8*60*60):
            break

    for i in range(len(features_selected)):
        features_selected[i] += 1

    print("Best features: ", features_selected)
        
    return features_selected

def backward_selection():
    pass

def driver():
    pass

if __name__ == "__main__":
    # Import dataset
    data = np.loadtxt('calibration3.txt')
    
    # Extract features and classes
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i][1:])
        y.append(int(data[i][0]))

    features = forward_selection(x, y)