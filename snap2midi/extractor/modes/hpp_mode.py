from .oafv2_mode import _OAFV2Mode


class _HPPMode(_OAFV2Mode):
    """ 
        _HPPMode is a class for extracting
        the necessary data for training the
        HPPNet architecture.
    """
    def __init__(self, config):
        super().__init__(config)
