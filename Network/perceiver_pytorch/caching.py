from functools import wraps

from torch import nn


def cache_by_name_fn(f, name=None):
    cache = {}

    @wraps(f)
    def cached_fn(*args,name, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if name in cache:
            return cache[name]
        cache[name] = f(*args, **kwargs)
        return cache[name]

    return cached_fn

