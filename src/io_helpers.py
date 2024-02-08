import pandas as pd
from math import exp, log, floor
from random import random, randrange
from itertools import islice
from io import StringIO

def reservoir_sample(iterable, k=1):
    """
    Fast way to sample n rows from a large dataset.
    
    Select k items uniformly from iterable. Returns the whole population if there are k or fewer items.

    Originally from https://bugs.python.org/issue41311#msg373733
    discussed here https://stackoverflow.com/questions/22258491/read-a-small-random-sample-from-a-big-csv-file-into-a-pandas-data-frame
    """
    iterator = iter(iterable)
    values = list(islice(iterator, k))

    W = exp(log(random())/k)
    while True:
        # skip is geometrically distributed
        skip = floor( log(random())/log(1-W) )
        selection = list(islice(iterator, skip, skip+1))
        if selection:
            values[randrange(k)] = selection[0]
            W *= exp(log(random())/k)
        else:
            return values
        
def sample_file(filepath, k):
    with open(filepath, 'r') as f:
        header = next(f)
        result = [header] + reservoir_sample(f, k)  # Corrected function call
    df = pd.read_csv(StringIO(''.join(result)))
    return df