import numpy as np

labels = None
values = None

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
    global labels, values
    labels, values = read_file(file_name)
    print(find_nearest_neighbor(labels, values))

def find_nearest_neighbor(labels, values):
    num_correct = 0
    for i in range(len(values)):
        # Get the values and the actual label for the current point i
        vec_i = values[i]
        label_i = labels[i]
    
        nearest_neighbor_index = None
        nearest_neighbor_val = float('inf')

        for j in range(len(values)):
            if i == j:
                # Avoid repeating the same data point
                continue
            vec_j = values[j]
            distance = np.linalg.norm(vec_i - vec_j)
            if distance < nearest_neighbor_val:
                nearest_neighbor_index = j
                nearest_neighbor_val = distance
    
        predict_label = labels[nearest_neighbor_index]
        if predict_label == label_i:
            num_correct += 1
    
    return num_correct / len(values)
   

main()

