import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np

style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler

def make_plot(values, title, colour, xlab, ylab):
    plt.figure(figsize=(10, 5))
    plt.plot(values, color = colour)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid()
    #plt.show()

# normalize stock data
def normalize_data(data):
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data.values.reshape(-1,1))
    return data

# function to create train, validation, test data given stock data and sequence length

def load_data(inputdata, seq_len, val_percent):
    data = []
    
    # create sequences
    for i in range(len(inputdata) - seq_len): 
        data.append(inputdata[i: i + seq_len])
    
    # convert to arrays
    data = np.array(data);   # this has a shape of (4703, 20, 6)
    
    # size of validation, train and test set
    valid_size = int(np.round(val_percent*data.shape[0]));
    train_set_size = data.shape[0] - valid_size; 
    
    # choose all but the last for training 
    x_train = data[:train_set_size,:-1,:]
    # choose the last sequence as targets
    y_train = data[:train_set_size,-1,:]
    
    x_valid = data[train_set_size:train_set_size+valid_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_size,-1,:]
        
    return x_train, y_train, x_valid, y_valid


# function to get the next batch data
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array   
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

