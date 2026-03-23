class DeviceNotConnectedError(ConnectionError):
    def __init__(self,message="This device is not connected yet!!! Try connect() first"):
        self.message = message
        super.__init__(self.message)

class DeviceAlreadyConnectedError(ConnectionError):
    def __init__(self, message="This device is already connected!!!"):
        self.message = message
        super().__init__(self.message)