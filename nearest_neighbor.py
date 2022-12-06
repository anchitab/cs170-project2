import numpy as np
import random

labels = None
values = None
total_features = float('inf')

def read_file(file_name):
    labels = []
    values = []
    with open(file_name) as f:
        contents = f.readline().split()
        while contents or (not values):
            label = contents[0]
            vals = np.array([float(num) for num in contents[1:]])
            labels.append(label)
            values.append(vals)
            contents = f.readline().split()
    
    return labels, values

def main():
    print('Welcome to Anchita Bora\'s Feature Selection Algorithm')
    print('Type in the file name to test: ')
    file_name = input().strip()
    print('\n')
    print('Enter the number of the algorithm you want to run.\n')
    print('1) Forward Selection\n')
    print('2) Backward Elimination\n')
    algorithm = input().strip()
    if algorithm not in {'1', '2'}:
        print('Invalid algorithm input, Exiting.')
        return        
    global labels, values, total_features
    labels, values = read_file(file_name)
    total_features = len(values[0])

    print(f'This dataset has {total_features} features (not including the class attribute), with {len(values)} instances.\n')
    initial_accuracy = round(leave_one_out_accuracy(labels, values, set(range(1, total_features + 1))), 2)
    print(f'Running nearest neighbors with all {total_features} features, using \"leave-one-out\" accuracy, ' + 
   f'I get an accuracy of {initial_accuracy*100}%')

    if algorithm == '1':
        forward_selection()
    else:
        backward_elimination()

def leave_one_out_accuracy(labels, values, features, feature_to_add=None):
    num_correct = 0
    for i in range(len(values)):
        # Get the values and the actual label for the current point i
        vec_i = values[i]
        
        # Get the features in use out of vector i (instead of all features)
        vec_i = [vec_i[ind] for ind in range(len(vec_i)) if (ind+1 in features or ind+1 == feature_to_add)]
        vec_i = np.array(vec_i)

        label_i = labels[i]

        nearest_neighbor_index = None
        nearest_neighbor_val = float('inf')
        
        for j in range(len(values)):
            if i == j:
                # Avoid repeating the same data point
                continue
            vec_j = values[j]

            # Get the features in use for vector j
            vec_j = [vec_j[ind] for ind in range(len(vec_j)) if (ind+1 in features or ind+1 == feature_to_add)]
            vec_j = np.array(vec_j)

            # Euclidean distance between vec_i and vec_j
            distance = np.linalg.norm(vec_i - vec_j)
            if distance < nearest_neighbor_val:
                nearest_neighbor_index = j
                nearest_neighbor_val = distance
    
        predict_label = labels[nearest_neighbor_index]
        if predict_label == label_i:
            num_correct += 1
    
    # return random.random()
    return num_correct / len(values)

def forward_selection():
    feature_set = set()
    best_feature_set = None
    best_set_accuracy = float('-inf')

    while len(feature_set) < total_features:
        best_accuracy = float('-inf')
        best_feature = None
        for feature in range(1, total_features + 1):
            if feature in feature_set:
                continue

            new_accuracy = leave_one_out_accuracy(labels, values, feature_set, feature)
            print(f'Using feature(s) {feature_set.union({feature})}: accuracy is {round(new_accuracy*100, 2)}%')

            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                best_feature = feature

        if best_accuracy > best_set_accuracy:
            best_set_accuracy = best_accuracy
            best_feature_set = feature_set.union({best_feature})
    
        print(f'Best feature was {best_feature} with accuracy {round(best_accuracy*100, 2)}%')
        print('\n')
        feature_set = feature_set.union({best_feature})
    
    print(f'Finished search! Best feature set was {best_feature_set} with accuracy {round(best_set_accuracy * 100, 2)}%')

def backward_elimination():
    feature_set = set(range(1, total_features + 1))

    # Different from forward selection, have to check the accuracy of the full set first before eliminating any features
    best_feature_set = feature_set
    best_set_accuracy = leave_one_out_accuracy(labels, values, feature_set)
    print(f'Feature set {feature_set} has accuracy {round(best_set_accuracy*100, 2)}%')

    # Don't want to run the loop if only one feature left, we'd only be checking accuracy of an empty feature set!
    while len(feature_set) > 1:            
        best_accuracy = float('-inf')
        best_feature = None
        for feature in feature_set:
            new_accuracy = leave_one_out_accuracy(labels, values, feature_set - {feature})
            print(f'Using feature(s) {feature_set - {feature}}: accuracy is {round(new_accuracy*100, 2)}%')

            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                best_feature = feature

        if best_accuracy > best_set_accuracy:
            best_set_accuracy = best_accuracy
            best_feature_set = feature_set - {best_feature}
    
        print(f'Best feature to remove was {best_feature} with accuracy {round(best_accuracy * 100, 2)}%')
        print('\n')
        feature_set = feature_set - {best_feature}
    
    print(f'Finished search! Best feature set was {best_feature_set} with accuracy {round(best_set_accuracy * 100, 2)}%')


main()

