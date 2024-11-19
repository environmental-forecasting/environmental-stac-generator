def flatten_list(lst):
    """Flatten a list of lists (or tuples)"""
    return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) or isinstance(sublist, tuple) else [sublist])]
