import unittest
import pandas as pd

import sys
sys.path.append("..")
from Energy_Analytics import Wrapper
# from Energy_Analytics.Import_Data import *

class TestImport(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):

    # @classmethod
    # def tearDownClass(cls):
    #   pass

    # def setUp(self):
    #   pass

    # def tearDown(self):
    #   pass

    def test_import_csv(self):
        main_obj = Wrapper()
        main_obj.import_data(folder_name='../data/', head_row=[5,5,0])
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()