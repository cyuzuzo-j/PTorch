import resource

def get_maxrss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

print(get_maxrss())
