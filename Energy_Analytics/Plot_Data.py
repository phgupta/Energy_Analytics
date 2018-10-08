""" This script contains functions for displaying various plots.

Last modified: October 8 2018

Authors \n
@author Pranav Gupta <phgupta@ucdavis.edu>

"""

import pandas as pd
import matplotlib.pyplot as plt


class Plot_Data:

    """ This class contains functions for displaying various plots.
   
    Attributes
    ----------
    count    : int
        Keeps track of the number of figures.

    """

    # Static variable to keep count of number of figures
    count = 1

    def __init__(self, figsize=(18,5)):
        """ Constructor.

        Parameters
        ----------
        figsize : tuple
            Size of figure.

        """
        self.figsize = figsize


    def baseline_projection_plot(self, y_true, y_pred, time_period, model_name, adj_r2,
                                data, input_col, output_col, model):
        """ Create baseline and projection plots.

        Parameters
        ----------
        y_true      : pd.Series()
            Actual y values.
        y_pred      : np.ndarray
            Predicted y values.
        time_period : list(str)
            Baseline and projection periods.
        model_name  : str
            Optimal model's name.
        adj_r2      : float
            Adjusted R2 score of optimal model.
        data        : pd.Dataframe()
            Data containing real values.
        input_col   : list(str)
            Predictor column(s).
        output_col  : str
            Target column.
        model       : func
            Optimal model.

        Returns
        -------
        matplotlib.figure
            Baseline plot

        """

        # Baseline and projection plots
        fig = plt.figure(Plot_Data.count)
        
        # Number of plots to display
        nrows = len(time_period) / 2
        
        # Plot 1 - Baseline
        base_df = pd.DataFrame()
        base_df['y_true'] = y_true
        base_df['y_pred'] = y_pred
        ax1 = fig.add_subplot(nrows, 1, 1)
        base_df.plot(ax=ax1, figsize=self.figsize,
            title='Baseline Period ({}-{}). \nBest Model: {}. \nBaseline Adj R2: {}'.format(time_period[0], time_period[1], 
                                                                                model_name, adj_r2))

        # Display projection plots
        if len(time_period) > 2:
            num_plot = 2
            for i in range(2, len(time_period), 2):
                ax = fig.add_subplot(nrows, 1, num_plot)
                period = (slice(time_period[i], time_period[i+1]))
                project_df = pd.DataFrame()    
                project_df['y_true'] = data.loc[period, output_col]
                project_df['y_pred'] = model.predict(data.loc[period, input_col])
               
                # Set all negative values to zero since energy > 0
                project_df['y_pred'][project_df['y_pred'] < 0] = 0

                project_df.plot(ax=ax, figsize=self.figsize,
                    title='Projection Period ({}-{})'.format(time_period[i], time_period[i+1]))
                num_plot += 1
        fig.tight_layout()

        Plot_Data.count += 1
        return fig


if __name__ == '__main__':

    obj = Plot_Data()