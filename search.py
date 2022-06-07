import numpy as np

def neareast_neighbor():
    pass

def cross_validation():
    pass

def forward_selection():
    pass

def backward_selection():
    pass

def driver():
    pass

if __name__ == "__main__":
    # Import dataset
    data = np.loadtxt('calibration1.txt')
    
    # Extract features and classes
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i][1:])
        y.append(int(data[i][0]))

    