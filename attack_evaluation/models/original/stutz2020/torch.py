import torch


def expand_as(tensor, tensor_as):
    """
    Expands the tensor using view to allow broadcasting.
    :param tensor: input tensor
    :type tensor: torch.Tensor or torch.autograd.Variable
    :param tensor_as: reference tensor
    :type tensor_as: torch.Tensor or torch.autograd.Variable
    :return: tensor expanded with singelton dimensions as tensor_as
    :rtype: torch.Tensor or torch.autograd.Variable
    """

    view = list(tensor.size())
    for i in range(len(tensor.size()), len(tensor_as.size())):
        view.append(1)

    return tensor.view(view)


class View(torch.nn.Module):
    """
    Simple view layer.
    """

    def __init__(self, *args):
        """
        Constructor.
        :param args: shape
        :type args: [int]
        """

        super(View, self).__init__()

        self.shape = args

    def forward(self, input):
        """
        Forward pass.
        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return input.view(self.shape)


class Flatten(torch.nn.Module):
    """
    Flatten module.
    """

    def forward(self, input):
        """
        Forward pass.
        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return input.view(input.shape[0], -1)


class Clamp(torch.nn.Module):
    """
    Wrapper for clamp.
    """

    def __init__(self, min=0, max=1):
        """
        Constructor.
        """

        super(Clamp, self).__init__()

        self.min = min
        """ (float) Min value. """

        self.max = max
        """ (float) Max value. """

    def forward(self, input):
        """
        Forward pass.
        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return torch.clamp(torch.clamp(input, min=self.min), max=self.max)


class Scale(torch.nn.Module):
    """
    Simply scaling layer, mainly to allow simple saving and loading.
    """

    def __init__(self, shape):
        """
        Constructor.
        :param shape: shape
        :type shape: [int]
        """

        super(Scale, self).__init__()

        self.weight = torch.nn.Parameter(torch.zeros(shape))  # min
        self.bias = torch.nn.Parameter(torch.ones(shape))  # max

    def forward(self, input):
        """
        Forward pass.
        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return expand_as(self.weight, input) + torch.mul(expand_as(self.bias, input) - expand_as(self.weight, input),
                                                         input)


class Normalize(torch.nn.Module):
    """
    Normalization layer to be learned.
    """

    def __init__(self, n_channels):
        """
        Constructor.
        :param n_channels: number of channels
        :type n_channels: int
        """

        super(Normalize, self).__init__()

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = torch.nn.Parameter(torch.ones(n_channels))
        self.bias = torch.nn.Parameter(torch.zeros(n_channels))

    def forward(self, input):
        """
        Forward pass.
        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return (input - self.bias.view(1, -1, 1, 1)) / self.weight.view(1, -1, 1, 1)
