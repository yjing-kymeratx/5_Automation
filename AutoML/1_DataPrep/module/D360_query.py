## ----------------- D360 data query -----------------
def _use_d360api(my_query_id=3539, user_name="yjing@kymeratx.com", tokenFile='yjing_D360.token'):
    from d360api import d360api
    # Create API connection to the PROD server
    my_d360 = d360api(provider="https://10.3.20.47:8080")  # PROD environment
    user_name = user_name
    tokenFile = tokenFile
    
    with open(tokenFile, 'r') as ofh:
        service_token = ofh.readlines()[0]

    # Authenticate connection using service token
    print(f"\tRun D360 query on ID {my_query_id}")
    my_d360.authenticate_servicetoken(servicetoken=service_token, user=user_name)
    query_result = my_d360.download_query_results(query_id=my_query_id)
    return query_result


## ----------------- folder checker -----------------
def _folderChecker(my_folder=None):
    import os
    if my_folder is None:
        my_folder = os.getcwd()
        print(f"\t<my_folder> is None, data will be saved to the current work directory")

    else:
        if not os.path.isdir(my_folder):
            my_folder = os.getcwd()
            print(f"\t<my_folder> is not exist, data will be saved to the current work directory")
        else:
            my_folder = my_folder
    return my_folder


## ----------------- data downlowd -----------------
def dataDownload(my_query_id=3539, user_name="yjing@kymeratx.com", tokenFile='yjing_D360.token', outputFolder=None):
    ## run the query using d360api
    result = _use_d360api(my_query_id=my_query_id, user_name=user_name, tokenFile=tokenFile)
    print(f'\tAll data have been downloaded in file {result}')

    ## correct the output folder
    outputFolder = _folderChecker(outputFolder)
    output_fileName = f"{outputFolder}/{result}"
    
    ## Move file to the correct folder    
    import shutil
    shutil.move(result, output_fileName)
    print(f"\tMove the downloaded file ./{output_fileName} to {output_fileName}")
    
    return output_fileName
