import unittest
import pandas as pd

import sys
sys.path.append("..")
from Energy_Analytics import Wrapper

class TestImport(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    #     main_obj = Wrapper()

    # @classmethod
    # def tearDownClass(cls):
    #   pass

    def setUp(self):
      self.main_obj = Wrapper()

    # def tearDown(self):
    #   pass

    def test_import_csv(self):
        
        with self.assertRaises(Exception) as context:
            self.main_obj.import_data(folder_name='../data/', head_row=[-1,5,0])
           
        with self.assertRaises(Exception) as context:         
            self.main_obj.import_data(folder_name='../data/', head_row=[-1,5,0,1,2,3])

        with self.assertRaises(SystemError) as context:
            self.main_obj.import_data(file_name='blah.csv', folder_name=['blah', 'blah'])

        with self.assertRaises(SystemError) as context:
            self.main_obj.import_data(file_name=['blah1.csv', 'blah2.csv'], folder_name=['blah1', 'blah2'])

        with self.assertRaises(Exception) as context:
            self.main_obj.import_data(file_name='blah.txt', folder_name='../')

        with self.assertRaises(SystemError) as context:
            self.main_obj.import_data(folder_name=['one', 'two'])


if __name__ == '__main__':
    unittest.main()