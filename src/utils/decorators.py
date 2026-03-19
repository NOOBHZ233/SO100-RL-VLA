from functools import wraps
from vvrobot.utils.errors import DeviceNotConnectError , DeviceAlreadyConnectError

#call the wrapper() before the input func be used
def check_if_not_connected(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_connected:
            raise DeviceNotConnectError
        return func(self, *args, **kwargs)
    
    return wrapper

def check_if_already_connected(func):
    @wraps(func)
    def wrapper(self,*args, **kwargs):
        if self.is_connected:
            raise DeviceAlreadyConnectError
    
    return wrapper