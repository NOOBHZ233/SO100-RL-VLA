from functools import wraps
from project.utils.errors import DeviceNotConnectedError , DeviceAlreadyConnectedError

#call the wrapper() before the input func be used
def check_if_not_connected(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_connected:
            raise DeviceNotConnectedError
        return func(self, *args, **kwargs)
    
    return wrapper

def check_if_already_connected(func):
    @wraps(func)
    def wrapper(self,*args, **kwargs):
        if self.is_connected:
            raise DeviceAlreadyConnectedError
    
    return wrapper