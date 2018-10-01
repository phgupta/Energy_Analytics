""" Interface for InfluxDB. """

import configparser
import pandas as pd
import numpy as np
from influxdb import InfluxDBClient
from influxdb import DataFrameClient


def transform_to_dict(s, tags):
    """ Transforms list to dictionary.

    Parameters
    ----------
    s       : list
        List of values
    tags    : list
        List of keys

    Returns
    -------
    dict
        Dictionary where the keys are passed in as a list and the values are obtained from the apply function as a row
    
    """
    dic = {}
    for tag in tags:
        dic[tag] = s[tag]
    return dic


class Influx_Dataframe_Client(object):

    """ This class is the interface for InfluxDB.

    Attributes
    ----------
    host                : str
        Host name.
    port                : str
        Port number.
    username            : str
        Username of account.
    password            : str
        Password of account.
    database            : str
        Database name.
    use_ssl             : bool
        Use SSL.
    verify_ssl_is_on    : bool
        Verifies is SSL is active.
    client              : str
        Creates an instance of InfluxDBClient.
    df_client           : str
        Creates an instance of DataFrameClient.
    data                : pd.DataFrame()
        Dataframe storing data.

    """

    #Connection details
    host                = ""
    port                = ""
    username            = ""
    password            = ""
    database            = ""
    use_ssl             = False
    verify_ssl_is_on    = False
    client              = None
    df_client           = None
    data                = None


    def __init__(self, config_file, db_section=None):
        """ Constructor reads credentials from config file and establishes a connection.

        Parameters
        ----------
        config_file     : str
            Configuration file name.
        db_section      : str
            Database section.

        """

        # read from config file
        Config = configparser.ConfigParser()
        Config.read(config_file)
        if db_section != None:
            self.db_config = Config[db_section]
        else:
            self.db_config = Config["DB_config"]

        self.host = self.db_config.get("host")
        self.username = self.db_config.get("username")
        self.password = self.db_config.get("password")
        self.database = self.db_config.get("database")
        self.protocol = self.db_config.get("protocol")
        self.port = self.db_config.get("port")
        self.use_ssl = self.db_config.getboolean("use_ssl")
        self.verify_ssl_is_on = self.db_config.getboolean("verify_ssl_is_on")
        self.__make_client()


    def __make_client(self):
        """ Setup client for both InfluxDBClient and DataFrameClient.
        
        DataFrameClient is for queries and InfluxDBClient is for writes.
        This function is not necessary for the user.

        """

        self.client = InfluxDBClient(host=self.host, port=self.port,
                    username=self.username, password=self.password,
                    database=self.database,ssl=self.use_ssl, verify_ssl=self.verify_ssl_is_on)
        self.df_client = DataFrameClient(host=self.host, port=self.port,
                    username=self.username, password=self.password,
                    database=self.database,ssl=self.use_ssl, verify_ssl=self.verify_ssl_is_on)


    def __build_json(self,data, tags, fields, measurement):
        """ Builds json dictionary list out of dataframe given in the format expected by InfluxDBClient.

        Both tags and fields need to be lists which include the columns in the dataframe that are going to be included in the tags
        and fields dictionary.
        This function is not necessary for the user.

        Parameters
        ----------
        data            : pd.DataFrame()
            Dataframe containing meter data.
        tags            : str
            Tags of the data.
        fields          : str
            Fields of the data.
        measurement     : str
            Measurement.

        Returns
        -------
        json
            JSON dictionary list.

        """

        data['measurement'] = measurement
        data["tags"] = data.loc[:,tags].apply(transform_to_dict, tags=tags, axis=1)
        data["fields"] = data.loc[:,fields].apply(transform_to_dict, tags=fields, axis=1)
        json = data[["measurement","time", "tags", "fields"]].to_dict("records")
        return json


    def __post_to_DB(self,json,database=None):
        """ Sends json dictionary list to specified database to InfluxDBClient.

        This function is necessary for the user.

        Parameters
        ----------
        json        : json
            JSON dictionary.
        database    : str
            Database name.

        Returns
        -------
        InfluxDBClient.write_points instance
           Data type returned by instance of InfluxDBClient.

        """
        ret = self.client.write_points(json,database=database,batch_size=16384)
        return ret


    def expose_influx_client(self):
        """ Expose InfluxDBClient to user so they can utilize all functions of InfluxDBClient if functionality is not provided by
        Influx_Dataframe_Client module.

        Returns
        -------
        InfluxDBClient
            Class attribute.

        """
        return self.client


    def expose_data_client(self):
        """ Expose DataFrameClient to user so they can utilize all functions of DataFrameClient if functionality is not provided by
        Influx_Dataframe_Client module.

        Returns
        -------
        DataFrameClient
            Class attribute.

        """
        return self.df_client


    def write_dataframe(self,data,tags,fields,measurement,database=None):
        """ Write a dataframe to the specified measurement.

        Parameters
        ----------
        data            : pd.DataFrame()
            Dataframe containing meter data.
        tags            : list(str)
            Tags of the data.
        fields          : list(str)
            Fields of the data.
        measurement     : str
            Measurement.
        database        : str
            Database name.

        Returns
        -------
        self.__post_to_DB()
            Calls and returns another class function.

        """

        #set default database
        if (database == None):
            database = self.database


        if 'time' not in data.columns: #check to see if the time column is present
            data.index.name = 'time' #change the index to name to time
            data = data.reset_index() # give seqeuential index to dataframe

        #Turn dataframe into correct json format as described in beginning comments
        json = self.__build_json(data,tags,fields,measurement)

        ret = self.__post_to_DB(json,database)
        return ret


    def write_csv(self,csv_fileName,tags,fields,measurement,database=None):
        """ Upload csv file data to database.

        Parameters
        ----------
        csv_filename    : str
            Name of csv file.
        tags            : list(str)
            Tags of the data.
        fields          : list(str)
            Fields of the data.
        measurement     : str
            Measurement.
        database        : str
            Database name. Defaults to the one specified by client.

        Returns
        -------
        self.write_dataframe()
            Calls and returns another class function.

        """

        #set default database
        if (database == None):
            database = self.database

        data = pd.read_csv(csv_fileName)
        ret = self.write_dataframe(data,tags,fields,measurement,database)
        return ret


    def write_json(self,json,database=None):
        """ Upload json data to database.

        Parameters
        ----------
        json            : dict
            JSON data to upload.
        database        : str
            Database name. Defaults to the one specified by client.   : 

        """

        #set default database
        if (database == None):
            database = self.database

        #check to see if json is a list of dictionaries or a single dictionary
        if isinstance(json, list):
            ret = self.__post_to_DB(json,database)
        else:
            json = [json]
            ret = self.__post_to_DB(json,database)

        return ret


    def list_DB(self):
        """ List all the names of the databases on the InfluxDB server.
        
        Returns
        -------
        list
            List of all the names of the databases on the InfluxDB server.

        """
        list_to_return = []
        DB_dict_list = self.client.get_list_database()

        for x in range(len(DB_dict_list)):
            list_to_return.append(DB_dict_list[x]['name'])

        return list_to_return


    def list_retention_policies(self):
        """ List all the names and its retention policies of the databases on the InfluxDB server.
        
        Returns
        -------
        dict
            Key: Name of Database, Value:  Associated retention policy.

        """

        DB_list = self.list_DB()
        dict_list = []
        for x in range(len(DB_list)):
            temp_dict = {}
            temp_dict[DB_list[x]] = self.client.get_list_retention_policies(DB_list[x])
            dict_list.append(temp_dict)
        
        return dict_list


    def query_data(self,query):
        """ Query InfluxDB.

        Parameters
        ----------
        query   : str
            Query in Influx Query Language.

        Returns
        -------
        pd.DataFrame()
            Data returned
        Sends the specified query string to the specified database using
        InfluxDBClient the query must be in Influx Query Language
        """
        df = self.df_client.query(query, database='wifi_data8',chunked=True, chunk_size=256)
        return df


    def query(self, query, use_database = None):
        '''
        Sends the specified query string to the specified database using the
        DataframeClient the query must be in Influx Query Language returns a
        dataframe
        '''
        query_result = self.client.query(query, database=use_database)
        return query_result.raw


    def show_meta_data(self, database, measurement):
        '''
        Returns a list of TAG KEYS for specified measurement in specified database
        Equivalent query is below
        SHOW TAG KEYS FROM "MEASUREMENT_ARGUMENT"
        '''

        result_list = []
        #generate query string and make query
        query_string = 'SHOW TAG KEYS FROM ' +'\"' + measurement + "\""
        query_result = self.client.query(query_string, database=database)
        #add all of the tag values into a list to be returned
        #query result is a generator
        for temp_dict in query_result.get_points():
            result_list.append(temp_dict['tagKey'])
        return result_list


    def get_meta_data(self,database, measurement,tag):
        '''
        Returns a list of TAG VALUES for specified measurement in specified database
        Equivalent query is below
        SHOW TAG VALUES FROM "MEASUREMENT_ARGUMENT" WITH KEY IN = "TAG_ARGUMENT"
        '''
        result_list = []
        #generate query string and make query
        query_string = 'SHOW TAG VALUES FROM ' + '\"' + measurement + '\"' + 'WITH KEY = \"' + tag + '\"'
        query_result = self.client.query(query_string, database=database)

        #add all of the tag values into a list to be returned
        #query result is a generator
        for temp_dict in query_result.get_points():
            result_list.append(temp_dict['value'])

        return result_list


    def get_meta_data_time_series(self,database, measurement, tags,start_time=None,end_time=None):
        '''
        Returns tags along with the time stamps
        '''

        #get all data with from measurement
        df = self.specific_query(database,measurement,start_time=start_time,end_time=end_time)
        return df[tags]


    def specific_query(self,database,measurement,fields=None,start_time=None,end_time=None,tags=None,values=None,groupList=None,groupTime=None):
        '''
        This function returns a dataframe with the results of the specified query
        the query is built using the parameters provided by the user and
        formatted into Influx Query Language. All fields are optional except the
        database and measurement parameter. This function always returns a
        dataframe even if the response has no results
        '''
        tag_string = ""
        time_string = ""
        group_string = ""
        df = {}
        #Create base query with fields and measurement
        query_string = "SELECT "
        if (fields == None):
            query_string = query_string + '* '
        else:
            for x in range(len(fields)):
                if (x > 0):
                    query_string = query_string + " ,"
                query_string = query_string + "\"" + fields[x] + "\""
        query_string = query_string + " FROM \"" + measurement + "\""

        #Create time portion of query if it is specified
        if (start_time != None or end_time != None ):
            if (start_time != None):
                #Must have a start_time for our query
                #Check to see format of time that was specified
                time_string = time_string + "time > "
                if type(end_time) == str:
                    time_string = time_string + "\'" + start_time + '\''
                if(type(end_time) == int):
                    time_string = time_string + str(start_time)

            if (end_time != None):
                #Must have a end_time for our query
                #Check to see format of time that was specified
                if (time_string != ""):
                    time_string = time_string + " AND "
                time_string = time_string + "time < "

                if type(end_time) == str:
                    time_string = time_string + "\'" + end_time + '\''

                if type(end_time) == int:
                    time_string = time_string + str(end_time)


        #Create tag portion of query if it is specified
        if (tags != None and values != None):
            try:
                if (len(tags) != len(values)):
                    print("Tags and values do not match raise exception later!")
                    raise BaseException
                else:
                    tag_string = ""
                    for x in range(len(tags)):
                        if (x > 0):
                            tag_string = tag_string + ' AND '
                        tag_string = tag_string + '\"' + tags[x] + "\" = \'" + values[x] + "\'"
            except BaseException:
                print("Tags and values do not match")
                return pd.DataFrame()
        if (groupList != None):
            query_string = query_string + "GROUP BY"
            for x in range(len(groupList)):
                if (x > 0):
                    query_string = query_string + ","
                if (groupList[x] == "time"):
                    query_string = query_string + "time(" + groupTime + ")"
                else:
                    query_string = query_string + "\""+groupList[x]+"\""

        #Add optional parts of query
        if (time_string != "" or tag_string != ""):
            query_string = query_string + " WHERE "
            if (time_string != ""):
                query_string = query_string + time_string
            if (tag_string != ""):
                if (time_string != ""):
                    query_string = query_string + " AND "
                query_string = query_string + tag_string
        if (group_string != ""):
            query_string = query_string + group_string

        print(query_string)

        df = self.df_client.query(query_string, database=database,chunked=True, chunk_size=256)

        if (measurement in df):
            return df[measurement]
        else:
            #Must have an empty result make empty dataframe
            df = pd.DataFrame()
        return df


    def delete_based_on_time(self,database,measurement,start_time=None,end_time=None):
        '''
        Delete data from measurement. If no time is specified then all data will
        be deleted from the measurement.
        '''
        time_string = ""
        query_string = "DELETE FROM %s "%measurement

        if (start_time != None):
            #Must have a start_time for our query
            #Check to see format of time that was specified
            time_string = time_string + "time > "
            if type(end_time) == str:
                time_string = time_string + "\'" + start_time + '\''
            if type(end_time) == int:
                time_string = time_string + str(start_time)

        if (end_time != None):
            #Must have a end_time for our query
            #Check to see format of time that was specified
            if (time_string != ""):
                time_string = time_string + " AND "
            time_string = time_string + "time < "

            if type(end_time) == str:
                time_string = time_string + "\'" + end_time + '\''

            if type(end_time) == int:
                time_string = time_string + str(end_time)

        if time_string != "":
            query_string = query_string + " WHERE "
            if (time_string != ""):
                query_string = query_string + time_string

        # print(query_string)
        df = self.df_client.query(query_string, database=self.database,chunked=True, chunk_size=256)
