import logging


class MyLogger(logging.Logger):
    def __init__(self, level):
        super().__init__(__name__)
        self.level = level
        self.setLevel(self.level)

    def set_file_handler(self, filename, level=None):
        formatter_file = logging.Formatter("%(asctime)s: %(message)s")
        fh = logging.FileHandler(filename=filename)
        fh.setLevel(level if level else self.level)
        fh.setFormatter(formatter_file)
        self.addHandler(fh)

    def set_stream_handler(self, level=None):
        formatter_stream = logging.Formatter("%(asctime)s: %(message)s")
        sh = logging.StreamHandler()
        sh.setLevel(level if level else self.level)
        sh.setFormatter(formatter_stream)
        self.addHandler(sh)