import torch.nn as nn

class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function
        # x shape: batch*seq_len， embedding_size  也就是说用整批的输出来预测有没有switch
        out = self.linear(x)
        return out

