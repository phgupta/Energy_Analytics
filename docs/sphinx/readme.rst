README
======
This is the official documentation for Energy_Analytics - a data analytics tool for building energy meter data.


Requirements
------------
1. Anaconda (Python 3)
2. Python 3.6

For the full list of requirements, check environment.yml


Setup
-----
1. Install the requirements.
2. Clone the repository.
3. Run ``conda env create -f environment.yml`` This creates a new python environment called "mortar".
4. Run ``conda activate mortar` (windows) or`` source activate mortar` (macOS/Linux) to start the environment shell.


Structure
---------
The primary purpose of the library is to create baselines of building energy meter data and calculate the cost & energy savings post retrofit.
Energy_Analytics/Energy_Analytics/ contains 9 files,

1. Import_Data.py
	This script contains two classes, Import_Data & Import_MDAL to extract data from csv files and MDAL respectively. Note - In order to extract data from MDAL, you need to run the mortar environment.
2. Clean_Data.py
	This script contains functions for cleaning the data, such as outlier detection, removing out-of-bounds data, resampling, interpolation...
3. Preprocess_Data.py
	This script contains functions for processing the cleaned data, such as adding time features (year, month, day, time-of-day, day-of-week), standardizing & normalizing data...
4. Model_Data.py
	This script contains functions for modeling the processed data with different Machine Learning algorithms like linear, lasso, ridge regression, random forest... and selecting the one with the best fit.
5. Plot_Data.py
	This script contains functions for creating plots.
6. Wrapper.py
	This script is a wrapper around the above scripts. User should use this to conduct data analysis. ADD EXPLANATION & USE CASES.
7. Main.ipynb
	This is a jupyter notebook that demonstrates how to use the library.
8. input.json
	This is a sample json file that the user can use to modify the parameters.
9. sample.json
	This is a sample output of the library.


Documentation
-------------
You can find the complete documentation at - https://phgupta.github.io/Energy_Analytics/