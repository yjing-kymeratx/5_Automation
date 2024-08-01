import chardet    # for <determine_encoding>


def determine_encoding(dataFile):
    # Step 1: Open the CSV file in binary mode
    with open(dataFile, 'rb') as f:
        data = f.read()
    
    # Step 2: Detect the encoding using the chardet library
    encoding_result = chardet.detect(data)

    # Step 3: Retrieve the encoding information
    encoding = encoding_result['encoding']

    # Step 4: Print/export the detected encoding information
    # print("Detected Encoding:", encoding)
    return encoding