# Data imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""This script is used to compare the train data from one scenario with the train data of another scenario. 

In this case the purpose is to compare the baseline with a selected disruption period."""

def compare_operations(train, train_selected, baseline_name = 'Baseline',
                       scenario_name = 'Selected period', plots = True)