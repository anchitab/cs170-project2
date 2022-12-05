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
    file_name = 'test_data.txt'
    global labels, values, total_features
    labels, values = read_file(file_name)
    total_features = len(values[0])

    # print(leave_one_out_accuracy(labels, values, list(range(1, len(values[0])+1))))
    forward_elimination()

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

            distance = np.linalg.norm(vec_i - vec_j)
            if distance < nearest_neighbor_val:
                nearest_neighbor_index = j
                nearest_neighbor_val = distance
    
        predict_label = labels[nearest_neighbor_index]
        if predict_label == label_i:
            num_correct += 1
    
    return random.random()
    # return num_correct / len(values)

def forward_elimination():
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
            print(f'Using feature(s) {feature_set.union({feature})}: accuracy is {new_accuracy*100}%')

            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                best_feature = feature

        if best_accuracy > best_set_accuracy:
            best_set_accuracy = best_accuracy
            best_feature_set = feature_set.union({best_feature})
    
        print(f'Best feature was {best_feature} with accuracy {best_accuracy * 100}%')
        feature_set = feature_set.union({best_feature})
    
    print(f'Finished search! Best feature set was {best_feature_set} with accuracy {best_set_accuracy * 100}%')

main()

