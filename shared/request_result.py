class RequestResult:
    def __init__(self, status, data, error: bool=False):
        self.status = status
        self.error: bool = error
        self.data = data
