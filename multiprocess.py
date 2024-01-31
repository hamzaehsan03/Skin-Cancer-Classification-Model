from multiprocessing import Pool
'''
generic parallelisation function
function = function to be processed in parallel
arguments = argument tuples
processors = number of parallel processes to use
'''
def parallel_process(function, arguments, processors=None):
    
    with Pool(processes=processors) as pool:
        results = pool.starmap(function, arguments)
    return results
