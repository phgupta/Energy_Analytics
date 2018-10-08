""" This script contains functions for displaying various plots.

Last modified: October 7 2018

Authors \n
@author Pranav Gupta <phgupta@ucdavis.edu>

"""

# from Energy_Analytics import Model_Data
from Model_Data import *

class Plot_Data(Model_Data):

    """ This class contains functions for displaying various plots.
   
    Attributes
    ----------
    count    : int
        Keeps track of the number of figures.

    """

    # Static variable to keep count of number of figures
    count = 1


    def __init__(self):
        """ Constructor """
        pass


if __name__ == '__main__':

    obj = Plot_Data()