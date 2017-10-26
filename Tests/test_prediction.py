import os
import numpy as np
import sys

# directory where project is located
#project_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/LSTM_Reduced"
project_path = os.getcwd() + "/LSTM_Reduced"


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
    
    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
    
    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


# test_capitalize.py

def capital_case(x):
    return x.capitalize()

def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'

def test_prediction():
    with cd(project_path):
        sys.path.insert(0, project_path)
        from prediction import main as perform_test_prediction
        prediction_output = perform_test_prediction()
    assert np.sum(prediction_output[:10,0,0] - [-2.85755336, -2.42378372, -2.02809878, -1.66715797, -1.33677999, -1.03306286, -0.75441993, -0.49803106, -0.2612441 , -0.04188509])<1e-6



def test_differential():
    with cd(project_path):
#        sys.path.insert(0, project_path)
        from functions import lorenz as lorenz
    dudt1 = lorenz(0, [1,1,0], 1.0, 2.0, 3.0)
    dudt2 = lorenz(0, [1,0,1], 1.0, 2.0, 3.0)
    dudt3 = lorenz(0, [0,1,1], 1.0, 2.0, 3.0)
    assert (np.sum(dudt1 - [ 0.,  1.,  1.])<1e-6 and np.sum(dudt2 - [ -1.,  1.,  -3.])<1e-6 and np.sum(dudt3 - [ 1.,  -1.,  -3.])<1e-6)




















