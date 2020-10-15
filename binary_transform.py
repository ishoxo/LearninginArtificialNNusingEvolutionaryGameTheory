import numpy as np


def transform_data(MNIST_dataframe):
    """
    :param MNIST_dataframe: dataframe of MNIST data reduced to 3 components
    :return: MNIST dataframe where each component has been approximated using 3 bits
            and the corresponding labels
    """
    comp1 = MNIST_dataframe['component1'].to_numpy()
    comp2 = MNIST_dataframe['component2'].to_numpy()
    comp3 = MNIST_dataframe['component3'].to_numpy()
    #label = MNIST_dataframe['label'].to_numpy()
    # find range and min values for each component
    ma1 = max(comp1)
    mi1 = min(comp1)
    ma2 = max(comp2)
    mi2 = min(comp2)
    ma3 = max(comp3)
    mi3 = min(comp3)


    range1 = ma1 - mi1
    range2 = ma2 - mi2
    range3 = ma3 - mi3
    ranges = [range1, range2, range3]
    mins = [mi1, mi2, mi3]

    Row_list = []
    # Iterate over each row
    for index, rows in MNIST_dataframe.iterrows():
        # Create list for the current row
        my_list = [rows.component1, rows.component2, rows.component3, rows.new_label]
        # append the list to the final list
        Row_list.append(my_list)

    binary_row_list = []
    row_labels = []
    for i in range(len(MNIST_dataframe)):
        bin_row = []
        data_point = Row_list[i]
        for j in range(len(data_point) - 1):
            data_range = ranges[j]
            comp = data_point[j]
            comp = comp - mins[j]
            if comp > (data_range/2):
                bin_row.append(1)
            else:
                bin_row.append(0)
            if (comp % (data_range/2)) > (data_range/4):
                bin_row.append(1)
            else:
                bin_row.append(0)
            if (comp % (data_range/2)) % (data_range/4) > (data_range/8):
                bin_row.append(1)
            else:
                bin_row.append(0)
        binary_row_list.append(np.asarray(bin_row))
        row_labels.append(data_point[-1])

    return binary_row_list, row_labels

def transform_data_multiclass(MNIST_dataframe):
    """
    :param MNIST_dataframe: MNIST dataframe witb continous values for components
    :return: MNIST dataframe with each compoonent approximated by 3 bits.
    """
    comp1 = MNIST_dataframe['component1'].to_numpy()
    comp2 = MNIST_dataframe['component2'].to_numpy()
    comp3 = MNIST_dataframe['component3'].to_numpy()
    #label = MNIST_dataframe['label'].to_numpy()

    ma1 = max(comp1)
    mi1 = min(comp1)
    ma2 = max(comp2)
    mi2 = min(comp2)
    ma3 = max(comp3)
    mi3 = min(comp3)


    range1 = ma1 - mi1
    range2 = ma2 - mi2
    range3 = ma3 - mi3
    ranges = [range1, range2, range3]
    mins = [mi1, mi2, mi3]



    Row_list = []
    # Iterate over each row
    for index, rows in MNIST_dataframe.iterrows():
        # Create list for the current row
        my_list = [rows.component1, rows.component2, rows.component3, rows.label]
        # append the list to the final list
        Row_list.append(my_list)

    binary_row_list = []
    row_labels = []
    for i in range(len(MNIST_dataframe)):
        bin_row = []
        data_point = Row_list[i]
        for j in range(len(data_point) - 1):
            data_range = ranges[j]
            comp = data_point[j]
            comp = comp - mins[j]
            if comp > (data_range/2):
                bin_row.append(1)
            else:
                bin_row.append(0)
            if (comp % (data_range/2)) > (data_range/4):
                bin_row.append(1)
            else:
                bin_row.append(0)
            if (comp % (data_range/2)) % (data_range/4) > (data_range/8):
                bin_row.append(1)
            else:
                bin_row.append(0)
        binary_row_list.append(np.asarray(bin_row))
        row_labels.append(data_point[-1])

    return binary_row_list, row_labels




