from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset as Dataset
from binary_transform import transform_data_multiclass
import numpy as np
import pickle
from restricted_NOR_networks import restricted_NOR_network
from restricted_NAND_networks import restricted_NAND_network
from IR_network_nomirror import mirror_network
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from network_evaluation_functions import convert_to_det
# Initialise the different types of network
rn0 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn1 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn2 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn3 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn4 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn5 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn6 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn7 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn8 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn9 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])

rna0 = restricted_NAND_network([9, 9, 9, 6, 3, 2, 1], [9])
rna1 = restricted_NAND_network([9, 9, 9, 6, 3, 2, 1], [9])
rna2 = restricted_NAND_network([9, 9, 9, 6, 3, 2, 1], [9])
rna3 = restricted_NAND_network([9, 9, 9, 6, 3, 2, 1], [9])
rna4 = restricted_NAND_network([9, 9, 9, 6, 3, 2, 1], [9])
rna5 = restricted_NAND_network([9, 9, 9, 6, 3, 2, 1], [9])
rna6 = restricted_NAND_network([9, 9, 9, 6, 3, 2, 1], [9])
rna7 = restricted_NAND_network([9, 9, 9, 6, 3, 2, 1], [9])
rna8 = restricted_NAND_network([9, 9, 9, 6, 3, 2, 1], [9])
rna9 = restricted_NAND_network([9, 9, 9, 6, 3, 2, 1], [9])

network_zero = mirror_network(neurons_in_each_layer=[9, 9, 6, 3, 1],
                               connections_in_each_layer=[3, 3, 3, 3, 3],
                               input_size=[9])

network_one = mirror_network(neurons_in_each_layer=[9, 9, 6, 3, 1],
                               connections_in_each_layer=[3, 3, 3, 3, 3],
                               input_size=[9])

network_two = mirror_network(neurons_in_each_layer=[9, 9, 6, 3, 1],
                               connections_in_each_layer=[3, 3, 3, 3, 3],
                               input_size=[9])

network_three = mirror_network(neurons_in_each_layer=[9, 9, 6, 3, 1],
                               connections_in_each_layer=[3, 3, 3, 3, 3],
                               input_size=[9])

network_four = mirror_network(neurons_in_each_layer=[9, 9, 6, 3, 1],
                               connections_in_each_layer=[3, 3, 3, 3, 3],
                               input_size=[9])

network_five = mirror_network(neurons_in_each_layer=[9, 9, 6, 3, 1],
                               connections_in_each_layer=[3, 3, 3, 3, 3],
                               input_size=[9])

network_six = mirror_network(neurons_in_each_layer=[9, 9, 6, 3, 1],
                               connections_in_each_layer=[3, 3, 3, 3, 3],
                               input_size=[9])

network_seven = mirror_network(neurons_in_each_layer=[9, 9, 6, 3, 1],
                               connections_in_each_layer=[3, 3, 3, 3, 3],
                               input_size=[9])

network_eight = mirror_network(neurons_in_each_layer=[9, 9, 6, 3, 1],
                               connections_in_each_layer=[3, 3, 3, 3, 3],
                               input_size=[9])

network_nine = mirror_network(neurons_in_each_layer=[9, 9, 6, 3, 1],
                               connections_in_each_layer=[3, 3, 3, 3, 3],
                               input_size=[9])




# open strategies from training
with open('strategies/NOR_strat_80.pkl', 'rb') as f:
    strats = pickle.load(f)

with open('strategies/NAND_strat_80.pkl', 'rb') as f:
    strats2 = pickle.load(f)

with open('strategies/IR_strat_80.pkl', 'rb') as f:
    strats3 = pickle.load(f)


#determine accuracy of networks to break ties; get final train scores for each network
with open('strategies/NAND_train_80.pkl', 'rb') as f:
    NAND_scores = pickle.load(f)
NAND_final_train_scores = []
for i in range(len(NAND_scores)):
    ts = NAND_scores[i]
    NAND_final_train_scores.append(ts[-1])

with open('strategies/NOR_train_80.pkl', 'rb') as f:
    NOR_train_scores = pickle.load(f)
NOR_final_train_scores = []
for i in range(len(NOR_train_scores)):
    class_train_scores = NOR_train_scores[i]
    NOR_final_train_scores.append(class_train_scores[-1])


with open('strategies/IRnm_train_80.pkl', 'rb') as f:
    NOR_train_scores = pickle.load(f)
IR_final_train_scores = []
for i in range(len(IR_final_train_scores)):
    class_train_scores = IR_final_train_scores[i]
    IR_final_train_scores.append(class_train_scores[-1])


#load strategies
NOR_networks = [rn0, rn1, rn2, rn3, rn4, rn5, rn6, rn7, rn8, rn9]
NAND_networks = [rna0, rna1, rna2, rna3, rna4, rna5, rna6, rna7, rna8, rna9]
IR_networks = [network_zero, network_one, network_two, network_three, network_four, network_five,
                  network_six, network_seven, network_eight, network_nine]
for i in range(len(NOR_networks)):
    network = NOR_networks[i]
    network.load_mixed_strategy(strats[i])

for i in range(len(NAND_networks)):
    network = NAND_networks[i]
    network.load_mixed_strategy(strats2[i])
for i in range(len(IR_networks)):
    network = IR_networks[i]
    network.load_mixed_strategy(strats3[i])


#convert mixed strategy networks to deterministic networks
IR_networks_det = []
NOR_networks_det = []
NAND_networks_det = []
for i in range(len(IR_networks)):
    network1 = IR_networks[i]
    network2 = NOR_networks[i]
    network3 = NAND_networks[i]
    network1 = convert_to_det(network1)
    network2 = convert_to_det(network2)
    network3 = convert_to_det(network3)
    IR_networks_det.append(network1)
    NOR_networks_det.append(network2)
    NAND_networks_det.append(network3)


#create test dataset
MNIST_data = pd.read_csv('TSNE_all_untouched')

MNIST_test = MNIST_data[60000:]

class MNIST_set(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        X_p = self.X[item]
        y_p = self.y[item]
        return X_p, y_p


X_test, y_test = transform_data_multiclass(MNIST_test)
test_dataset = MNIST_set(X_test, y_test)






def test_ensemble(networks1, networks2, networks3, scores1, scores2, scores3, dataset):
    """
    returns confusion matrices and accuracy for ensemble classifiers (IR, NOR and NAND networks)
    """
    y_true = []
    y_pred = []
    accuracy = 0
    for id, data in enumerate(dataset):
        if id % 1000 == 0:
            print('Sample: ', id)
        X, y = data
        y = int(y)
        outputs = [0] * 10
        for r in range(1):
            for i in range(len(networks1)):
                network = networks1[i]
                output, strategies, _ = network.forward(X)
                outputs[i] += output[0] * scores1[i]
            for j in range(len(networks2)):
                network = networks2[j]
                output, strategies, _ = network.forward(X)
                outputs[j] += output[0] * scores2[j]

            for k in range(len(networks3)):
                network = networks2[k]
                output, strategies, _ = network.forward(X)
                outputs[k] += output[0] * scores3[k]

        if y == np.argmax(outputs):
            accuracy += 1

        y_pred.append(np.argmax(outputs))
        y_true.append(y)


    C = confusion_matrix(y_true, y_pred)
    C = C / C.astype(np.float).sum(axis=1)
    df_cm = pd.DataFrame(C, index=[i for i in "0123456789"],
                             columns=[i for i in "0123456789"])
    plt.figure(figsize=(10, 7))
    plt.title('Confusion matrix for Ensemble Classifiers (pure strategies')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    sn.heatmap(df_cm, annot=True, cmap="OrRd")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print('Accuracy:', accuracy/len(dataset))
    return accuracy/len(dataset)

test_ensemble(NOR_networks_det, IR_networks_det, NAND_networks_det, NOR_final_train_scores, IR_final_train_scores, NAND_final_train_scores, test_dataset)



