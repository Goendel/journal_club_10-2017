import os
import numpy as np
import sys
project_path = "/Users/pantelisvlachas/Documents/PhD/journal_club_travis/LSTM_Reduced"
sys.path.insert(0, project_path)

# directory where project is located
#project_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/LSTM_Reduced"







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
        from prediction import main as perform_test_prediction
        prediction_output = perform_test_prediction()
    assert np.sum(prediction_output[:10,0,0] - [-2.85755336, -2.42378372, -2.02809878, -1.66715797, -1.33677999, -1.03306286, -0.75441993, -0.49803106, -0.2612441 , -0.04188509])<1e-6













