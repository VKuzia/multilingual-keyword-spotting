from torch import nn


class LambdaLayer(nn.Module):
    def __init__(self, lambda_):
        super(LambdaLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return self.lambda_(x)
