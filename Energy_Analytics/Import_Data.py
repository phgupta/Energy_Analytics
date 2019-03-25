""" This script imports data from csv files, MDAL and returns a dataframe. 

Note
----
Last modified: March 25 2019

1. CSV - If only folder is specified and no filename, all csv's will be read in sorted order by name.
2. CSV - Doesn't handle cases when user provides
    - file_name of type str and folder_name of type list(str)
    - file_name and folder_name both of type list(str)


To Do
-----
    1. Figure out parameter/return types of functions (do a search on "???")


Authors
-------
- Marco Pritoni <marco.pritoni@gmail.com>
- Jacob Rodriguez  <jbrodriguez@ucdavis.edu>
- Pranav Gupta <phgupta@ucdavis.edu>

"""

import os
import glob
import pymortar
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
from collections import defaultdict


class Import_Data:

    """ This class imports data from csv files. """

    def __init__(self):
        """ Constructor: Store the imported data. """
        self.data = pd.DataFrame()


    def import_csv(self, file_name='*', folder_name='.', head_row=0, index_col=0, convert_col=True, concat_files=False):
        """ Imports csv file(s) and stores the result in data.
            
        Note
        ----
        1. If folder exists out of current directory, folder_name should contain correct regex
        2. Assuming there's no file called "\*.csv"

        Parameters
        ----------
        file_name       : str
            CSV file to be imported. Defaults to '\*', i.e. all csv files in the folder.
        folder_name     : str
            Folder where file resides. Defaults to '.', i.e. current directory.
        head_row        : int
            Skips all rows from 0 to head_row-1
        index_col       : int
            Skips all columns from 0 to index_col-1
        convert_col     : bool
            Convert columns to numeric type
        concat_files    : bool
            Appends data from files to result dataframe

        """

        # Import a specific or all csv files in folder
        if isinstance(file_name, str) and isinstance(folder_name, str):
            try:
                self.data = self._load_csv(file_name, folder_name, head_row, index_col, convert_col, concat_files)
            except Exception as e:
                raise e

        # Import multiple csv files in a particular folder.
        elif isinstance(file_name, list) and isinstance(folder_name, str):

            for i, file in enumerate(file_name):
                if isinstance(head_row, list):
                    _head_row = head_row[i]
                else:
                    _head_row = head_row

                if isinstance(index_col, list):
                    _index_col = index_col[i]
                else:
                    _index_col = index_col

                try:
                    data_tmp = self._load_csv(file, folder_name, _head_row, _index_col, convert_col, concat_files)
                    if concat_files:
                        self.data = self.data.append(data_tmp, ignore_index=False, verify_integrity=False)
                    else:
                        self.data = self.data.join(data_tmp, how="outer")
                except Exception as e:
                    raise e

        else:
            # Current implementation can't accept,
            # 1. file_name of type str and folder_name of type list(str)
            # 2. file_name and folder_name both of type list(str)
            raise NotImplementedError("Filename and Folder name can't both be of type list.")


    def _load_csv(self, file_name, folder_name, head_row, index_col, convert_col, concat_files):
        """ Load single csv file.

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

        Returns
        -------
        pd.DataFrame()
            Dataframe containing csv data

        """

        # Denotes all csv files
        if file_name == "*":

            if not os.path.isdir(folder_name):
                raise OSError('Folder does not exist.')
            else:
                file_name_list = sorted(glob.glob(folder_name + '*.csv'))

                if not file_name_list:
                    raise OSError('Either the folder does not contain any csv files or invalid folder provided.')
                else:
                    # Call previous function again with parameters changed (file_name=file_name_list, folder_name=None)
                    # Done to reduce redundancy of code
                    self.import_csv(file_name=file_name_list, head_row=head_row, index_col=index_col,
                                    convert_col=convert_col, concat_files=concat_files)
                    return self.data

        else:
            if not os.path.isdir(folder_name):
                raise OSError('Folder does not exist.')
            else:
                path = os.path.join(folder_name, file_name)

                if head_row > 0:
                    data = pd.read_csv(path, index_col=index_col, skiprows=[i for i in range(head_row-1)])
                else:
                    data = pd.read_csv(path, index_col=index_col)

                # Convert time into datetime format
                try:
                    # Special case format 1/4/14 21:30
                    data.index = pd.to_datetime(data.index, format='%m/%d/%y %H:%M')
                except:
                    data.index = pd.to_datetime(data.index, dayfirst=False, infer_datetime_format=True)

        # Convert all columns to numeric type
        if convert_col:
            # Check columns in dataframe to see if they are numeric
            for col in data.columns:
                # If particular column is not numeric, then convert to numeric type
                if data[col].dtype != np.number:
                    data[col] = pd.to_numeric(data[col], errors="coerce")

        return data


class Import_Mortar(Import_Data):

    """ This class queries data from Mortar.

    Note
    ----
    Set the evironment variables - $MORTAR_API_USERNAME & $MORTAR_API_PASSWORD.

    For Mac,
    1. vi ~/.bash_profile
    2. Add at the end of file,
        1. export $MORTAR_API_USERNAME=username
        2. export $MORTAR_API_PASSWORD=password
    3. source ~/.bash_profile


    """

    def __init__(self):
        """ Constructor. """

        self.client = pymortar.Client({})


    @staticmethod
    def convert_to_utc(time):
        """ Convert time to UTC

        Parameters
        ----------
        time    : str
            Time to convert. Has to be of the format '2016-01-01T00:00:00-08:00'.

        Returns
        -------
        str
            UTC timestamp.

        """

        # time is already in UTC
        if 'Z' in time:
            return time
        else:
            time_formatted = time[:-3] + time[-2:]
            dt = datetime.strptime(time_formatted, '%Y-%m-%dT%H:%M:%S%z')
            dt = dt.astimezone(timezone('UTC'))
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


    def get_meter(self, site, start, end, point_type="Green_Button_Meter", agg=pymortar.MEAN, window='15m'):
        """ Get meter data from Mortar.

        Parameters
        ----------
        site            : list(str)
            List of sites.
        start           : str
            Start date - 'YYYY-MM-DDTHH:MM:SSZ'
        end             : str
            End date - 'YYYY-MM-DDTHH:MM:SSZ'
        point_type      : str
            Type of data, i.e. Green_Button_Meter, Building_Electric_Meter...
        agg             : pymortar aggregation object
            Values include pymortar.MEAN, pymortar.MAX, pymortar.MIN, 
        pymortar.COUNT, pymortar.SUM, pymortar.RAW (the temporal window parameter is ignored)
        window          : str
            Size of the moving window.
        
        Returns
        -------
        pd.DataFrame(), defaultdict(list)
            Meter data, dictionary that maps meter data's columns (uuid's) to sitenames.

        """

        # CHECK: Does Mortar take in UTC or local time? 
        # Convert time to UTC
        start = self.convert_to_utc(start)
        end = self.convert_to_utc(end)

        query_meter = "SELECT ?meter WHERE { ?meter rdf:type/rdfs:subClassOf* brick:" + point_type + " };"

        # Get list of sites for meter data
        resp_meter = self.client.qualify([query_meter])

        if resp_meter.error:
            raise RuntimeError(resp_meter.error)

        # Define the view of meters (metadata)
        meter = pymortar.View(
            name="view_meter",
            sites=site,
            definition=query_meter
        )

        # Define the meter timeseries stream
        data_view_meter = pymortar.DataFrame(
            name="data_meter", # dataframe column name
            aggregation=agg,
            window=window,
            timeseries=[
                pymortar.Timeseries(
                    view="view_meter",
                    dataVars=["?meter"]
                )
            ]
        )

        # Define timeframe
        time_params = pymortar.TimeParams(
            start=start,
            end=end
        )

        # Form the full request object
        request = pymortar.FetchRequest(
            sites=site,
            views=[meter],
            dataFrames=[data_view_meter],
            time=time_params
        )

        # Fetch data from request
        response = self.client.fetch(request)

        # resp_meter = (url, uuid, sitename)
        resp_meter = response.query('select * from view_meter')

        # Map's uuid's to the site names
        map_uuid_sitename = defaultdict(list)
        for (url, uuid, sitename) in resp_meter:
            map_uuid_sitename[uuid].append(sitename)

        return response['data_meter'], map_uuid_sitename
            

    def get_weather(self, site, start, end, point_type='Weather_Temperature_Sensor', agg=pymortar.MEAN, window='15m'):
        """ Get weather (OAT) data from Mortar.

        Parameters
        ----------
        site            : list(str)
            List of sites.
        start           : str
            Start date - 'YYYY-MM-DDTHH:MM:SSZ'
        end             : str
            End date - 'YYYY-MM-DDTHH:MM:SSZ'
        point_type      : str
            Type of point, i.e. Weather_Temperature_Sensor...
        agg             : pymortar aggregation object
            Values include pymortar.MEAN, pymortar.MAX, pymortar.MIN, 
        pymortar.COUNT, pymortar.SUM, pymortar.RAW (the temporal window parameter is ignored)
        window          : str
            Size of the moving window.
        
        Returns
        -------
        pd.DataFrame(), defaultdict(list)
            OAT data, dictionary that maps meter data's columns (uuid's) to sitenames.

        """

        # CHECK: Does Mortar take in UTC or local time? 
        # Convert time to UTC
        start = self.convert_to_utc(start)
        end = self.convert_to_utc(end)

        query_oat = "SELECT ?t WHERE { ?t rdf:type/rdfs:subClassOf* brick:" + point_type + " };"

        # Get list of sites for OAT data
        resp_oat = self.client.qualify([query_oat])

        if resp_oat.error:
            raise RuntimeError(resp_oat.error)

        # Define the view of meters (metadata)
        oat = pymortar.View(
            name="view_oat",
            sites=site,
            definition=query_oat
        )

        # Define the meter timeseries stream
        data_view_oat = pymortar.DataFrame(
            name="data_oat", # dataframe column name
            aggregation=agg,
            window=window,
            timeseries=[
                pymortar.Timeseries(
                    view="view_oat",
                    dataVars=["?t"]
                )
            ]
        )

        # Define timeframe
        time_params = pymortar.TimeParams(
            start=start,
            end=end
        )

        # Form the full request object
        request = pymortar.FetchRequest(
            sites=site,
            views=[oat],
            dataFrames=[data_view_oat],
            time=time_params
        )

        # Fetch data from request
        response = self.client.fetch(request)

        # resp_meter = (url, uuid, sitename)
        resp_oat = response.query('select * from view_oat')

        # Map's uuid's to the site names
        map_uuid_sitename = defaultdict(list)
        for (url, uuid, sitename) in resp_oat:
            map_uuid_sitename[uuid].append(sitename)

        return response['data_oat'], map_uuid_sitename


    def get_tstat(self, site, start, end, agg=pymortar.MAX, window='1m'):
        """ Get tstat data from Mortar.

        Parameters
        ----------
        site            : list(str)
            List of sites.
        start           : str
            Start date - 'YYYY-MM-DDTHH:MM:SSZ'
        end             : str
            End date - 'YYYY-MM-DDTHH:MM:SSZ'
        agg             : pymortar aggregation object
            Values include pymortar.MEAN, pymortar.MAX, pymortar.MIN, 
        pymortar.COUNT, pymortar.SUM, pymortar.RAW (the temporal window parameter is ignored)
        window          : str
            Size of the moving window.
        
        Returns
        -------
        pd.DataFrame()
            Dataframe containing tstat data for all sites.

        """

        # CHECK: Does Mortar take in UTC or local time? 
        # Convert time to UTC
        start = self.convert_to_utc(start)
        end = self.convert_to_utc(end)

        query_tstat = "SELECT ?tstat ?room ?zone ?state ?temp ?hsp ?csp WHERE { \
            ?tstat bf:hasLocation ?room . \
            ?zone bf:hasPart ?room . \
            ?tstat bf:hasPoint ?state . \
            ?tstat bf:hasPoint ?temp . \
            ?tstat bf:hasPoint ?hsp . \
            ?tstat bf:hasPoint ?csp . \
            ?zone rdf:type/rdfs:subClassOf* brick:Zone . \
            ?tstat rdf:type/rdfs:subClassOf* brick:Thermostat . \
            ?state rdf:type/rdfs:subClassOf* brick:Thermostat_Status . \
            ?temp  rdf:type/rdfs:subClassOf* brick:Temperature_Sensor  . \
            ?hsp   rdf:type/rdfs:subClassOf* brick:Supply_Air_Temperature_Heating_Setpoint . \
            ?csp   rdf:type/rdfs:subClassOf* brick:Supply_Air_Temperature_Cooling_Setpoint . \
        };"

        # Get list of sites for tstat data
        resp_tstat = self.client.qualify([query_tstat])

        if resp_tstat.error:
            raise RuntimeError(resp_tstat.error)

        # Define the view of tstat (metadata)
        tstat = pymortar.View(
            name="view_tstat",
            sites=site,
            definition=query_tstat
        )

        # Define the meter timeseries stream
        data_view_tstat = pymortar.DataFrame(
            name="data_tstat", # dataframe column name
            aggregation=agg,
            window=window,
            timeseries=[
                pymortar.Timeseries(
                    view="view_tstat",
                    dataVars=["?state", "?temp", "?hsp", "?csp"]
                )
            ]
        )

        # Define timeframe
        time_params = pymortar.TimeParams(
            start=start,
            end=end
        )

        # Form the full request object
        request = pymortar.FetchRequest(
            sites=site,
            views=[tstat],
            dataFrames=[data_view_tstat],
            time=time_params
        )

        # Fetch data from request
        response = self.client.fetch(request)

        # Final dataframe containing all sites' data
        df_result = pd.DataFrame()
        
        tstat_df = response['data_tstat']
        tstats = [tstat[0] for tstat in response.query("select tstat from view_tstat")]
        error_df_list = []

        for i, tstat in enumerate(tstats):
            
            q = """
                SELECT state_uuid, temp_uuid, hsp_uuid, csp_uuid, room, zone, site
                FROM view_tstat
                WHERE tstat = "{0}";
            """.format(tstat)
        
            res = response.query(q)
            if not res:
                continue

            state_col, iat_col, hsp_col, csp_col, room, zone, site = res[0]
            df = tstat_df[[state_col, iat_col, hsp_col, csp_col]]
            
            # A single site has many tstat points. Adding site+str(i) distinguishes each of them.
            # CHECK: This can have a better naming scheme.
            df.columns = [site+str(i)+'_state',  site+str(i)+'_iat', site+str(i)+'_hsp', site+str(i)+'_csp']
            
            df_result = df_result.join(df, how='outer')

        return df_result
            

    def get_error_message(self, x, resample_minutes=60):
        """ Creates error message for a row of error_df (get_tstat())

        Parameters
        ----------
        x                   : row of pd.DataFrame()
            Pandas row.
        resample_minutes    : int
            Resampling minutes.
        
        Returns
        -------
        str
            Error message.

        """
        
        dt_format = "%Y-%m-%d %H:%M:%S"
        st = x.name
        st_str = st.strftime(dt_format)
        et_str = (st + timedelta(minutes=resample_minutes)).strftime(dt_format)
        site = x.site
        room = x.room
        zone = x.zone
        heat_percent = round(x.heat_percent, 2)
        cool_percent = round(x.cool_percent, 2)
        msg = "From {0} to {1}, zone: \'{2}\' in room: \'{3}\' at site: \'{4}\', was heating for {5}% of the time and cooling for {6}% of the time".format(
            st_str,
            et_str,
            zone,
            room,
            site,
            heat_percent,
            cool_percent
        )

        return msg
 

class Import_MDAL(Import_Data):

    """ This class imports data from MDAL. """

    def __init__(self):
        """ Constructor. """
        
        import dataclient
        self.m = dataclient.MDALClient("corbusier.cs.berkeley.edu:8088")


    @staticmethod
    def convert_to_utc(time):
        """ Convert time to UTC
        Parameters
        ----------
        time    : str
            Time to convert. Has to be of the format '2016-01-01T00:00:00-08:00'.
        Returns
        -------
        str
            UTC timestamp.
        """

        # time is already in UTC
        if 'Z' in time:
            return time
        else:
            time_formatted = time[:-3] + time[-2:]
            dt = datetime.strptime(time_formatted, '%Y-%m-%dT%H:%M:%S%z')
            dt = dt.astimezone(timezone('UTC'))
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


    def get_meter(self, site, start, end, point_type='Green_Button_Meter',
                  var="meter", agg='MEAN', window='24h', aligned=True, return_names=True):
        """ Get meter data from MDAL.
        Parameters
        ----------
        site            : str
            Building name.
        start           : str
            Start date - 'YYYY-MM-DDTHH:MM:SSZ'
        end             : str
            End date - 'YYYY-MM-DDTHH:MM:SSZ'
        point_type      : str
            Type of data, i.e. Green_Button_Meter, Building_Electric_Meter...
        var             : str
            Variable - "meter", "weather"...
        agg             : str
            Aggregation - MEAN, SUM, RAW...
        window          : str
            Size of the moving window.
        aligned         : bool
            ???
        return_names    : bool
            ???
        Returns
        -------
        (df, mapping, context)
            ???
        """

        # Convert time to UTC
        start = self.convert_to_utc(start)
        end = self.convert_to_utc(end)
    
        request = self.compose_MDAL_dic(point_type=point_type, site=site, start=start, end=end,
                                        var=var, agg=agg, window=window, aligned=aligned)
        resp = self.m.query(request)
        
        if return_names:
            resp = self.replace_uuid_w_names(resp)
        
        return resp


    def get_weather(self, site, start, end, point_type='Weather_Temperature_Sensor', 
                    var="weather", agg='MEAN', window='24h', aligned=True, return_names=True):
        """ Get weather data from MDAL.
        Parameters
        ----------
        site            : str
            Building name.
        start           : str
            Start date - 'YYYY-MM-DDTHH:MM:SSZ'
        end             : str
            End date - 'YYYY-MM-DDTHH:MM:SSZ'
        point_type      : str
            Type of data, i.e. Green_Button_Meter, Building_Electric_Meter...
        var             : str
            Variable - "meter", "weather"...
        agg             : str
            Aggregation - MEAN, SUM, RAW...
        window          : str
            Size of the moving window.
        aligned         : bool
            ???
        return_names    : bool
            ???
        Returns
        -------
        (df, mapping, context)
            ???
        """

        # Convert time to UTC
        start = self.convert_to_utc(start)
        end = self.convert_to_utc(end)

        request = self.compose_MDAL_dic(point_type=point_type, site=site, start=start, end=end,
                                        var=var, agg=agg, window=window, aligned=aligned)
        resp = self.m.query(request)

        if return_names:
            resp = self.replace_uuid_w_names(resp)

        return resp


    def get_tstat(self, site, start, end, var="tstat_temp", agg='MEAN', window='24h', aligned=True, return_names=True):
        """ Get thermostat data from MDAL.
        Parameters
        ----------
        site            : str
            Building name.
        start           : str
            Start date - 'YYYY-MM-DDTHH:MM:SSZ'
        end             : str
            End date - 'YYYY-MM-DDTHH:MM:SSZ'
        var             : str
            Variable - "meter", "weather"...
        agg             : str
            Aggregation - MEAN, SUM, RAW...
        window          : str
            Size of the moving window.
        aligned         : bool
            ???
        return_names    : bool
            ???
        Returns
        -------
        (df, mapping, context)
            ???
        """

        # Convert time to UTC
        start = self.convert_to_utc(start)
        end = self.convert_to_utc(end)
    
        point_map = {
            "tstat_state" : "Thermostat_Status", 
            "tstat_hsp" : "Supply_Air_Temperature_Heating_Setpoint", 
            "tstat_csp" : "Supply_Air_Temperature_Cooling_Setpoint", 
            "tstat_temp": "Temperature_Sensor" 
        }
        
        if isinstance(var,list):
            point_type = [point_map[point_type] for point_type in var] # list of all the point names using BRICK classes
        else:
            point_type = point_map[var] # single value using BRICK classes
        
        request = self.compose_MDAL_dic(point_type=point_type, site=site, start=start, end=end,
                                        var=var, agg=agg, window=window, aligned=aligned)
        resp = self.m.query(request)
        
        if return_names:
            resp = self.replace_uuid_w_names(resp)

        return resp


    def compose_MDAL_dic(self, site, point_type, 
                        start, end,  var, agg, window, aligned, points=None, return_names=False):
        """ Create dictionary for MDAL request.
        Parameters
        ----------
        site            : str
            Building name.
        start           : str
            Start date - 'YYYY-MM-DDTHH:MM:SSZ'
        end             : str
            End date - 'YYYY-MM-DDTHH:MM:SSZ'
        point_type      : str
            Type of data, i.e. Green_Button_Meter, Building_Electric_Meter...
        var             : str
            Variable - "meter", "weather"...
        agg             : str
            Aggregation - MEAN, SUM, RAW...
        window          : str
            Size of the moving window.
        aligned         : bool
            ???
        return_names    : bool
            ???
        Returns
        -------
        (df, mapping, context)
            ???
        """

        # Convert time to UTC
        start = self.convert_to_utc(start)
        end = self.convert_to_utc(end)
    
        request = {} 
        
        # Add Time Details - single set for one or multiple series
        request['Time'] = {
            'Start': start,
            'End': end,
            'Window': window,
            'Aligned': aligned
                           }
        # Define Variables 
        request["Variables"] = {}
        request['Composition'] = []
        request['Aggregation'] = {}
        
        if isinstance(point_type, str): # if point_type is a string -> single type of point requested
            request["Variables"][var] =  self.compose_BRICK_query(point_type=point_type,site=site) # pass one point type at the time
            request['Composition'] = [var]
            request['Aggregation'][var] = [agg]
            
        elif isinstance(point_type, list): # loop through all the point_types and create one section of the brick query at the time

            for idx, point in enumerate(point_type): 
                request["Variables"][var[idx]] =  self.compose_BRICK_query(point_type=point,site=site) # pass one point type at the time
                request['Composition'].append(var[idx])
                
                if isinstance(agg, str): # if agg is a string -> single type of aggregation requested
                    request['Aggregation'][var[idx]] = [agg]
                elif isinstance(agg, list): # if agg is a list -> expected one agg per point
                    request['Aggregation'][var[idx]] = [agg[idx]]
        
        return request


    def compose_BRICK_query(self, point_type, site):
        """ Compose the BRICK query.
        Parameters
        ----------
        site            : str
            Building name.
        point_type      : str
            Type of data, i.e. Green_Button_Meter, Building_Electric_Meter...
        Returns
        -------
        dict
            BRICK query.
        """
    
        if point_type == "Green_Button_Meter" or point_type == 'Building_Electric_Meter':
            BRICK_query = {"Definition": """SELECT ?point ?uuid FROM %s WHERE {
                                                        ?point rdf:type brick:%s .
                                                        ?point bf:uuid ?uuid                
                                                                              };""" % (site,point_type)
                          }

        if point_type == "Weather_Temperature_Sensor":
            BRICK_query = {"Definition": """SELECT ?point ?uuid FROM %s WHERE {
                                                   ?point rdf:type/rdfs:subClassOf* brick:%s .
                                                   ?point bf:uuid ?uuid
                                                                            };""" % (site,point_type)
                          }
            
        if point_type in ["Thermostat_Status","Supply_Air_Temperature_Heating_Setpoint",
                          "Supply_Air_Temperature_Cooling_Setpoint","Temperature_Sensor"]: ##  "tstat_state","tstat_hsp","tstat_csp","tstat_temp": 
            BRICK_query = {"Definition": """SELECT ?point ?uuid ?equip FROM %s WHERE {
                                            ?point rdf:type/rdfs:subClassOf* brick:%s .
                                            ?point bf:uuid ?uuid .
                                            ?point bf:isPointOf ?equip .
                                            ?equip rdf:type brick:Thermostat };""" % (site,point_type)
                          }
        
        return BRICK_query


    def parse_context(self, context):
        """ Parse context.
        Parameters
        ----------
        context     : ???
            ???
        Returns
        -------
        pd.DataFrame()
            Pandas dataframe containing metadata.
        """
    
        metadata_table = pd.DataFrame(context).T
        return metadata_table


    def strip_point_name(self, col):
        """ Strip point name.
        Parameters
        ----------
        col     : ???
            ???
        Returns
        -------
        ???
            ???
        """
        return col.str.split("#", expand=True)[1]


    def get_point_name(self, context):
        """ Get point name.
        Parameters
        ----------
        context     : ???
            ???
        Returns
        -------
        ???
            ???
        """
        
        metadata_table = self.parse_context(context)
        return metadata_table.apply(self.strip_point_name, axis=1)


    def replace_uuid_w_names(self, resp):
        """ Replace the uuid's with names.
        Parameters
        ----------
        resp     : ???
            ???
        Returns
        -------
        ???
            ???
        """
        
        col_mapper = self.get_point_name(resp.context)["?point"].to_dict()
        resp.df.rename(columns=col_mapper, inplace=True)
        return resp