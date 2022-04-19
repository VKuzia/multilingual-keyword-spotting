from torch import nn


class LambdaLayer(nn.Module):
    """
    This Module is here for debug reasons :)
    Runs given lambda on its forward call and returns initial data.
    ! Lambda must not change any value in x
    """

    def __init__(self, lambda_):
        """Lambda must not change any value in data passed to it"""
        super(LambdaLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        self.lambda_(x)
        return x
