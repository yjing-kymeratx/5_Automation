#!/usr/bin/env python

##########################################################################
######################### 1. load the packages ###########################
##########################################################################
## ignore warning msg
import warnings
warnings.filterwarnings('ignore')

## import other packages
import os
import argparse
import pandas as pd
from d360api import d360api
from datetime import datetime

today = datetime.today().date().strftime('%Y-%m-%d')

##########################################################################
####################### 2. Build custom functions ########################
##########################################################################
## loading tokens from  token file
def loadToken(tokenFile='yjing_D360.token'):
    try:
        with open(tokenFile, 'r') as ofh:
            service_token = ofh.readlines()[0]
    except Exception as e:
        print(f'Can not load token file. {e}')
        service_token = None
    else:
        print(f'Successfully loading D360 API token from {tokenFile}')
    return service_token

##--------------------------------------------------------------
## loading tokens from  token file

def dataDownload(query_id=2905, user_name='yjing@kymeratx.com', service_token=None, provider='https://10.3.20.47:8080'):
    # Create API connection to the PROD server
    my_d360 = d360api(provider=provider)  # PROD environment
    
    # Authenticate connection using service token
    my_d360.authenticate_servicetoken(servicetoken=service_token, user=user_name)

    try:
        results = my_d360.download_query_results(query_id=query_id)
    except Exception as e:
        print(f'Can not download data. {e}')
        results = None
    else:
        print(f'All data have been downloaded in file {results}')
    return results

##--------------------------------------------------------------
def loadCSV(dataTableFileName, oFileName='data_raw.csv'):
    try:
        pd.read_csv(dataTableFileName)
    except Exception as e:
        print(f'Error: cannot read output file {dataTableFileName}; error msg: {e}')
        dataTable = None
    else:
        # dataTable = dataTable.reset_index(drop=True)
        # print(f'The downloaded data have data shape {dataTable.shape}')
        os.rename(dataTableFileName, oFileName)
        print(f'The downloaded data have been resaved to {oFileName}')
    return None


##########################################################################
####################### 3. define the main func ##########################
##########################################################################
def main():
    ## ------------------------ define the parser ------------------------
    parser = argparse.ArgumentParser(description='Auto-ML-Part-1: download assay data from D360')

    ## Add arguments
    parser.add_argument('-q', type=int, default=-1, help='the D360 query id of the search')
    parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
    parser.add_argument('-p', '--portal', type=str, default='https://10.3.20.47:8080', help='the IP and port of the D360 server PROD environment')
    parser.add_argument('-u', '--user', type=str, default='yjing@kymeratx.com', help='the user name of the search, should be Kymera email account')
    parser.add_argument('-t', '--token', type=str, default='/fsx/data/AUTOMATION/ML/tokens/yjing_D360.token', help='the token for access internal data through D360 API')
    parser.add_argument('-o', '--output', type=str, default='D360_api_pull_raw.csv', help='the file name of the output csv')

    ## Parse the arguments
    args = parser.parse_args()

    ## Use the arguments 
    QUERY_ID = args.q
    D360_PORT = args.portal
    USER_NAME = args.user
    TOKEN_FILE = args.token

    assert QUERY_ID > 0, "Please use -q to define the D360 query id. For example, use 2905 for ADMET property"
    
    ## ------------------------ load token ------------------------
    TOKEN_STR = loadToken(TOKEN_FILE)
    if TOKEN_STR is None:
        TOKEN_STR = loadToken(tokenFile='/home/yjing/data0/software/D360/yjing_D360.token')
    assert TOKEN_STR is not None, "Please check the token file"

    ## ------------------------ download data ------------------------
    dataFileName = dataDownload(query_id=QUERY_ID, user_name=USER_NAME, service_token=TOKEN_STR, provider=D360_PORT)

    ## ------------------------ load data ------------------------
    loadCSV(dataFileName)

##########################################################################
if(__name__ == "__main__"):
    main()

