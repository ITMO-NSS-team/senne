from senne.log import senne_logger


class RemoteSensingPreprocessor:
    """ Class for applying data preprocessing for remote sensing data during ensembling """

    def __init__(self):
        pass

    def smooth(self):
        """ Apply gaussian smoothing for images """
        raise NotImplementedError()
