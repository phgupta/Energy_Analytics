""" This script is a wrapper class around all the other modules - importing, cleaning, preprocessing and modeling the data.

Notes
-----
1. df.loc[(slice(None, None, None)), ...] is equivalent to "df.loc[:,...]"
2. df.resample(freq='h').mean() drops all non-float/non-int columns
3. os._exit(1) exits the program without calling cleanup handlers.

To Do,
1. Model
1.1 Add TimeSeriesSplit, ANN, SVM, Randomforest.
1.2. Add max_iter as a parameter.
1.3. Regression outputs negative values! Set predict = 0 in projected values (model_data/display_plots())
2. Wrapper
2.1. Run iterations on resampling frequency, adding time features (TOD, DOW, DOY...) !!!!
2.2. Add option to standardize/normalize data before fitting to model (Preprocess?)
2.3. Add Pearson's correlation coefficient.
2.4. Give user the option to run specific models.
3. All
3.1. Change SystemError to specific errors.
3.2. Ensure Python2.7 compatibility.
4. Cleanup
4.1. Run pylint on all files.
4.2. Documentation.
4.3. Structure code to publish to PyPI.

Authors,
Last modified: September 15 2018
@author Pranav Gupta <phgupta@ucdavis.edu>

"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from Energy_Analytics import Import_Data
from Energy_Analytics import Clean_Data
from Energy_Analytics import Preprocess_Data
from Energy_Analytics import Model_Data


class Wrapper:

    """ This class is a wrapper class around all the other modules - importing, cleaning, preprocessing and modeling the data.

    Attributes
    ----------
    figure_count    : int
        Keeps track of the number of iterations when searching for optimal model. Primarily used in search function.

    """

    # Static variable to keep count of number of iterations
    global_count = 1


    def __init__(self, results_folder_name='results'):
        """ Constructor.

        Initializes variables and creates results directory.

        Parameters
        ----------
        results_folder_name     : str
            Name of folder where results will reside

        """

        self.results_folder_name    = results_folder_name
        self.result                 = {}                    # Dictionary containing all the metrics
        self.best_metrics           = {}                    # Metrics of optimal model
        
        self.imported_data          = pd.DataFrame()
        self.cleaned_data           = pd.DataFrame()
        self.preprocessed_data      = pd.DataFrame()
        
        # Store UTC Time
        self.result['Time (UTC)'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        # Create results folder if it doesn't exist
        if not os.path.isdir(self.results_folder_name):
            os.makedirs(self.results_folder_name)


    def write_json(self):
        """ Dump data into json file """
        with open(self.results_folder_name + '/results-' + str(Wrapper.global_count) + '.json', 'a') as f:
            json.dump(self.result, f)


    def read_json(self, file_name):
        """ Read input json file.

        Notes
        -----
        The input json file should include ALL parameters.

        Parameters
        ----------
        file_name    : str
            Filename to be read.

        """

        if not isinstance(file_name, str) or not file_name.endswith('.json') or not os.path.isfile('./'+file_name):
            raise SystemError('File name should be a valid .json file residing in current directory.')


        with open(file_name) as f:

            input_json = json.load(f)

            import_json = input_json['Import']
            imported_data = self.import_data(file_name=import_json['File Name'], folder_name=import_json['Folder Name'],
                                            head_row=import_json['Head Row'], index_col=import_json['Index Col'],
                                            convert_col=import_json['Convert Col'], concat_files=import_json['Concat Files'])

            clean_json = input_json['Clean']
            cleaned_data = self.clean_data(imported_data, rename_col=clean_json['Rename Col'], drop_col=clean_json['Drop Col'],
                                            resample=clean_json['Resample'], freq=clean_json['Frequency'],
                                            interpolate=clean_json['Interpolate'], limit=clean_json['Limit'],
                                            method=clean_json['Method'], remove_na=clean_json['Remove NA'],
                                            remove_na_how=clean_json['Remove NA How'], remove_outliers=clean_json['Remove Outliers'],
                                            sd_val=clean_json['SD Val'], remove_out_of_bounds=clean_json['Remove Out of Bounds'],
                                            low_bound=clean_json['Low Bound'], high_bound=clean_json['High Bound'])

            preproc_json = input_json['Preprocess']
            preprocessed_data = self.preprocess_data(cleaned_data, cdh_cpoint=preproc_json['CDH CPoint'],
                                                    hdh_cpoint=preproc_json['HDH CPoint'], col_hdh_cdh=preproc_json['HDH CDH Calc Col'],
                                                    col_degree=preproc_json['Col Degree'], degree=preproc_json['Degree'],
                                                    year=preproc_json['Year'], month=preproc_json['Month'], week=preproc_json['Week'],
                                                    tod=preproc_json['Time of Day'], dow=preproc_json['Day of Week'], doy=preproc_json['Day of Year'],
                                                    var_to_expand=preproc_json['Variables to Expand'])

            model_json = input_json['Model']
            model_data = self.model(preprocessed_data, ind_col=model_json['Independent Col'], dep_col=model_json['Dependent Col'],
                                    time_period=model_json['Time Period'], exclude_time_period=model_json['Exclude Time Period'],
                                    alphas=model_json['Alphas'], cv=model_json['CV'], plot=model_json['Plot'], figsize=model_json['Fig Size'])

            self.write_json()


    # CHECK: Modify looping of time_freq
    def search(self, file_name, imported_data=None,
                resample_freq=['15T', 'h', 'd'], 
                time_freq = {
                    'year'  :   [True,  False,  False,  False,  False,  False],
                    'month' :   [False, True,   False,  False,  False,  False],
                    'week'  :   [False, False,  True,   False,  False,  False],
                    'tod'   :   [False, False,  False,  True,   False,  False],
                    'dow'   :   [False, False,  False,  False,  True,   False],
                    'doy'   :   [False, False,  False,  False,  False,  True]
                }):
        """ Run models on different data configurations.

        Notes
        -----
        The input json file should include ALL parameters.

        Parameters
        ----------
        file_name       : str
            Optional json file to read parameters.
        imported_data   : pd.DataFrame()
            Pandas Dataframe containing data.
        resample_freq   : list(str)
            Resampling frequencies to test.
        time_freq       : dict
            Time features to test.

        """

        # CSV Files
        if not imported_data:

            with open(file_name) as f:
                input_json = json.load(f)
                import_json = input_json['Import']
                imported_data = self.import_data(file_name=import_json['File Name'], folder_name=import_json['Folder Name'],
                                                head_row=import_json['Head Row'], index_col=import_json['Index Col'],
                                                convert_col=import_json['Convert Col'], concat_files=import_json['Concat Files'])

        with open(file_name) as f:
            
            input_json = json.load(f)

            for x in resample_freq:

                    clean_json = input_json['Clean']    # CHECK: Move line above for loop.
                    cleaned_data = self.clean_data(imported_data, rename_col=clean_json['Rename Col'], drop_col=clean_json['Drop Col'],
                                                resample=clean_json['Resample'], 
                                                freq=x,
                                                interpolate=clean_json['Interpolate'], limit=clean_json['Limit'],
                                                method=clean_json['Method'], remove_na=clean_json['Remove NA'],
                                                remove_na_how=clean_json['Remove NA How'], remove_outliers=clean_json['Remove Outliers'],
                                                sd_val=clean_json['SD Val'], remove_out_of_bounds=clean_json['Remove Out of Bounds'],
                                                low_bound=clean_json['Low Bound'], high_bound=clean_json['High Bound'])
                    
                    preproc_json = input_json['Preprocess']
                    preprocessed_data = self.preprocess_data(cleaned_data, cdh_cpoint=preproc_json['CDH CPoint'],
                                hdh_cpoint=preproc_json['HDH CPoint'], col_hdh_cdh=preproc_json['HDH CDH Calc Col'],
                                col_degree=preproc_json['Col Degree'], degree=preproc_json['Degree'],
                                year=preproc_json['Year'], month=preproc_json['Month'], week=preproc_json['Week'],
                                tod=preproc_json['Time of Day'], dow=preproc_json['Day of Week'], doy=preproc_json['Day of Year'],
                                var_to_expand=preproc_json['Variables to Expand'])

                    model_json = input_json['Model']
                    model_data = self.model(preprocessed_data, #global_count=Wrapper.global_count,
                        ind_col=model_json['Independent Col'], dep_col=model_json['Dependent Col'],
                        time_period=model_json['Time Period'], exclude_time_period=model_json['Exclude Time Period'],
                        alphas=model_json['Alphas'], cv=model_json['CV'], plot=model_json['Plot'], figsize=model_json['Fig Size'])

                    self.write_json()
                    Wrapper.global_count += 1


    def import_data(self, file_name='*', folder_name='.', head_row=0, index_col=0,
                    convert_col=True, concat_files=False, save_file=True):
        """ Imports csv file(s) and stores the result in self.imported_data.
            
        Note
        ----
        1. If folder exists out of current directory, folder_name should contain correct regex
        2. Assuming there's no file called "*.csv"

        Parameters
        ----------
        file_name       : str
            CSV file to be imported. Defaults to '*' - all csv files in the folder.
        folder_name     : str
            Folder where file resides. Defaults to '.' - current directory.
        head_row        : int
            Skips all rows from 0 to head_row-1
        index_col       : int
            Skips all columns from 0 to index_col-1
        convert_col     : bool
            Convert columns to numeric type
        concat_files    : bool
            Appends data from files to result dataframe
        save_file       : bool
            Specifies whether to save file or not. Defaults to True.

        Returns
        -------
        pd.DataFrame()
            Dataframe containing imported data.

        """
        
        # Create instance and import the data
        import_data_obj = Import_Data()
        import_data_obj.import_csv(file_name=file_name, folder_name=folder_name, 
                                head_row=head_row, index_col=index_col, 
                                convert_col=convert_col, concat_files=concat_files)
        
        # Store imported data in wrapper class
        self.imported_data = import_data_obj.data

        # Logging
        self.result['Import'] = {
            'Source': 'CSV', # import_data() supports only csv files currently
            'File Name': file_name,
            'Folder Name': folder_name,
            'Head Row': head_row,
            'Index Col': index_col,
            'Convert Col': convert_col,
            'Concat Files': concat_files
        }
        
        if save_file:
            f = self.results_folder_name + '/imported_data-' + str(Wrapper.global_count) + '.csv'
            self.imported_data.to_csv(f)
            self.result['Import']['Saved File'] = f
        else:
            self.result['Import']['Saved File'] = ''

        return self.imported_data


    def clean_data(self, data, rename_col=None, drop_col=None,
                    resample=True, freq='h',
                    interpolate=True, limit=1, method='linear',
                    remove_na=True, remove_na_how='any',
                    remove_outliers=True, sd_val=3,
                    remove_out_of_bounds=True, low_bound=0, high_bound=float('inf'),
                    save_file=True):
        """ Cleans dataframe according to user specifications and stores result in self.cleaned_data.

        Parameters
        ----------
        data                    : pd.DataFrame()
            Dataframe to be cleaned.
        rename_col              : list(str)
            List of new column names.
        drop_col                : list(str)
            Columns to be dropped.
        resample                : bool
            Indicates whether to resample data or not.
        freq                    : str
            Resampling frequency i.e. d, h, 15T... 
        interpolate             : bool
            Indicates whether to interpolate data or not.
        limit                   : int
            Interpolation limit.
        method                  : str
            Interpolation method.
        remove_na               : bool
            Indicates whether to remove NAs or not.
        remove_na_how           : str
            Specificies how to remove NA i.e. all, any...
        remove_outliers         : bool
            Indicates whether to remove outliers or not.
        sd_val                  : int
            Standard Deviation Value (specifices how many SDs away is a point considered an outlier)
        remove_out_of_bounds    : bool
            Indicates whether to remove out of bounds datapoints or not.
        low_bound               : int
            Low bound of the data.
        high_bound              : int
            High bound of the data.
        save_file       : bool
            Specifies whether to save file or not. Defaults to True.

        Returns
        -------
        pd.DataFrame()
            Dataframe containing cleaned data.

        """

        # Check to ensure data is a pandas dataframe
        if not isinstance(data, pd.DataFrame):
            raise SystemError('data has to be a pandas dataframe.')
        
        # Create instance and clean the data
        clean_data_obj = Clean_Data(data)
        clean_data_obj.clean_data(resample=resample, freq=freq, interpolate=interpolate,
                                limit=limit, remove_na=remove_na, remove_na_how=remove_na_how,
                                remove_outliers=remove_outliers, sd_val=sd_val,
                                remove_out_of_bounds=remove_out_of_bounds,
                                low_bound=low_bound, high_bound=high_bound)
        
        if rename_col:  # Rename columns of dataframe
            clean_data_obj.rename_columns(rename_col)
        if drop_col:    # Drop columns of dataframe
            clean_data_obj.drop_columns(drop_col)
        
        # Store cleaned data in wrapper class
        self.cleaned_data = clean_data_obj.cleaned_data

        # Logging
        self.result['Clean'] = {
            'Rename Col': rename_col,
            'Drop Col': drop_col,
            'Resample': resample,
            'Frequency': freq,
            'Interpolate': interpolate,
            'Limit': limit,
            'Method': method,
            'Remove NA': remove_na,
            'Remove NA How': remove_na_how,
            'Remove Outliers': remove_outliers,
            'SD Val': sd_val,
            'Remove Out of Bounds': remove_out_of_bounds,
            'Low Bound': low_bound,
            'High Bound': str(high_bound) if high_bound == float('inf') else high_bound
        }

        if self.imported_data.empty:
            self.result['Clean']['Source'] = '' # User provided their own dataframe, i.e. they did not use import_data()
        else:
            self.result['Clean']['Source'] = self.results_folder_name + '/imported_data-' + str(Wrapper.global_count) + '.csv'

        if save_file:
            f = self.results_folder_name + '/cleaned_data-' + str(Wrapper.global_count) + '.csv'
            self.cleaned_data.to_csv(f)
            self.result['Clean']['Saved File'] = f
        else:
            self.result['Clean']['Saved File'] = ''

        return self.cleaned_data


    def preprocess_data(self, data,
                        hdh_cpoint=65, cdh_cpoint=65, col_hdh_cdh='OAT',
                        col_degree=None, degree=None,
                        year=False, month=False, week=False, tod=False, dow=False, doy=False,
                        var_to_expand=None,
                        save_file=True):
        """ Preprocesses dataframe according to user specifications and stores result in self.preprocessed_data.

        Parameters
        ----------
        data            : pd.DataFrame()
            Dataframe to be preprocessed.
        hdh_cpoint      : int
            Heating degree hours. Defaults to 65.
        cdh_cpoint      : int
            Cooling degree hours. Defaults to 65.
        col_hdh_cdh     : str
            Column name which contains the outdoor air temperature.
        col_degree      : list(str)
            Column to exponentiate.
        degree          : list(str)
            Exponentiation degree.
        year            : bool
            Year.
        month           : bool
            Month.
        week            : bool
            Week.
        tod             : bool
            Time of Day.
        dow             : bool
            Day of Week.
        doy             : bool
            Day of Year.
        var_to_expand   : list(str)
            Variables to one-hot encode, e.g. var_to_expand=['tod', 'dow', 'year'].
        save_file       : bool
            Specifies whether to save file or not. Defaults to True.

        Returns
        -------
        pd.DataFrame()
            Dataframe containing preprocessed data.

        """

        # Check to ensure data is a pandas dataframe
        if not isinstance(data, pd.DataFrame):
            raise SystemError('data has to be a pandas dataframe.')
        
        # Create instance
        preprocess_data_obj = Preprocess_Data(data)
        preprocess_data_obj.add_degree_days(col=col_hdh_cdh, hdh_cpoint=hdh_cpoint, cdh_cpoint=cdh_cpoint)
        preprocess_data_obj.add_col_features(col=col_degree, degree=degree)
        preprocess_data_obj.add_time_features(year=year, month=month, week=week, tod=tod, dow=dow, doy=doy)
        preprocess_data_obj.create_dummies(var_to_expand=var_to_expand)
        
        # Store preprocessed data in wrapper class
        self.preprocessed_data = preprocess_data_obj.preprocessed_data

        # Logging
        self.result['Preprocess'] = {
            'HDH CPoint': hdh_cpoint,
            'CDH CPoint': cdh_cpoint,
            'HDH CDH Calc Col': col_hdh_cdh,
            'Col Degree': col_degree,
            'Degree': degree,
            'Year': year,
            'Month': month,
            'Week': week,
            'Time of Day': tod,
            'Day of Week': dow,
            'Day of Year': doy,
            'Variables to Expand': var_to_expand
        }

        if self.cleaned_data.empty:
            self.result['Preprocess']['Source'] = '' # User provided their own dataframe, i.e. they did not use cleaned_data()
        else:
            self.result['Preprocess']['Source'] = self.results_folder_name + '/cleaned_data-' + str(Wrapper.global_count) + '.csv'

        if save_file:
            f = self.results_folder_name + '/preprocessed_data-' + str(Wrapper.global_count) + '.csv'
            self.preprocessed_data.to_csv(f)
            self.result['Preprocess']['Saved File'] = f
        else:
            self.result['Preprocess']['Saved File'] = ''

        return self.preprocessed_data


    def model(self, data,
            ind_col=None, dep_col=None, time_period=[None,None], exclude_time_period=[None,None], 
            alphas=np.logspace(-4,1,30),
            cv=3, plot=True, figsize=None,
            custom_model_func=None):
        """ Split data into baseline and projection periods, run models on them and display metrics & plots.

        Parameters
        ----------
        data                    : pd.DataFrame()
            Dataframe to model.
        ind_col                 : list(str)
            Independent column(s) of dataframe. Defaults to all columns except the last.
        dep_col                 : str
            Dependent column of dataframe.
        time_period             : list(str)
            List of time periods to split the data into baseline and projection periods. It needs to have a start and an end date.
        exclude_time_period     : list(str)
            List of time periods to exclude for modeling.
        alphas                  : list(int)
            List of alphas to run regression on.
        cv                      : int
            Number of folds for cross-validation.
        plot                    : bool
            Specifies whether to save plots or not.
        figsize                 : tuple
            Size of the plots.
        custom_model_func       : function
            Model with specific hyper-parameters provided by user.

        Returns
        -------
        dict
            Metrics of the optimal/best model.

        """

        # Check to ensure data is a pandas dataframe
        if not isinstance(data, pd.DataFrame):
            raise SystemError('data has to be a pandas dataframe.')
        
        # Create instance
        model_data_obj = Model_Data(data, ind_col, dep_col, time_period, exclude_time_period, alphas, cv)

        # Split data into baseline and projection
        model_data_obj.split_data()
        
        # Logging
        self.result['Model'] = {
            'Independent Col': ind_col,
            'Dependent Col': dep_col,
            'Time Period': time_period,
            'Exclude Time Period': exclude_time_period,
            'Alphas': list(alphas),
            'CV': cv,
            'Plot': plot,
            'Fig Size': figsize,
            # 'Optimal Model': best_model_name
            # Add custom model func name?
        }

        # Runs all models on the data and returns optimal model
        best_model, metrics = model_data_obj.run_models()
        self.result['Model']['All Models'] = metrics

        # CHECK: Define custom model's parameter and return types in documentation.
        if custom_model_func:
            self.result['Model']['Custom Model\'s Metrics'] = model_data_obj.custom_model(custom_model_func)

        # Fit optimal model to data
        model_data_obj.best_model_fit(best_model)

        # Save metrics of optimal model
        self.best_metrics = model_data_obj.display_metrics()
        self.result['Model']['Optimal Model\'s Metrics'] = self.best_metrics

        if plot:
            fig1, fig2 = model_data_obj.display_plots(figsize)
            fig1.savefig(self.results_folder_name + '/acc_alpha-' + str(Wrapper.global_count) + '.png')
            fig2.savefig(self.results_folder_name + '/modeled_data-' + str(Wrapper.global_count) + '.png')

        if self.preprocessed_data.empty:
            self.result['Model']['Source'] = '' # User provided their own dataframe, i.e. they did not use preprocessed_data()
        else:
            self.result['Model']['Source'] = self.results_folder_name + '/preprocessed_data-' + str(Wrapper.global_count) + '.csv'
        
        return self.best_metrics


if __name__ == '__main__':
        
    ################ IMPORT DATA FROM CSV FILES #################
    def func(X, y):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        model = LinearRegression()
        model.fit(X, y)
        return model.predict(X)

    wrapper_obj = Wrapper()
    imported_data = wrapper_obj.import_data(folder_name='../../../../Desktop/LBNL/Data/', head_row=[5,5,0])
    cleaned_data = wrapper_obj.clean_data(imported_data, high_bound=9998,
                                    rename_col=['OAT','RelHum_Avg', 'CHW_Elec', 'Elec', 'Gas', 'HW_Heat'],
                                    drop_col='Elec')

    preprocessed_data = wrapper_obj.preprocess_data(cleaned_data, week=True, tod=True, var_to_expand=['tod','week'])

    wrapper_obj.model(preprocessed_data, dep_col='HW_Heat', alphas=np.logspace(-4,1,5), figsize=(18,5),
                   time_period=["2014-01","2014-12", "2015-01","2015-12", "2016-01","2016-12"],
                   cv=5,
                   exclude_time_period=['2014-06', '2014-07'],
                   custom_model_func=func)
    wrapper_obj.write_json()