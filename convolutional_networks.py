"""
Implements convolutional networks in PyTorch.
"""
from pickle import EMPTY_SET
from matplotlib.pyplot import color_sequences, colorbar
import torch
from ps2_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU



def hello_convolutional_networks():
    """
    This is the sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride = conv_param['stride']
        pad = conv_param['pad']
        
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        
        out = torch.zeros(N, F, H_out, W_out).to(x.dtype).to(x.device)
        
        padded_input = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        
        for batch in range(N):
          for filters in range(F):
              for height in range(0, H_out):
                  for width in range(0, W_out):
                      out[batch, filters, height, width] = torch.sum(
                          padded_input[batch, :, height * stride : height * stride + HH, width * stride : width * stride + WW] * w[filters]
                      ) + b[filters]

        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        
        x, w, b, conv_param = cache
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride = conv_param.get('stride', 1)
        pad = conv_param.get('pad', 0)

        W_out = 1 + (W + 2 * pad - WW) // stride
        H_out = 1 + (H + 2 * pad - HH) // stride
        x_padding = torch.nn.functional.pad(x, (pad, pad, pad, pad)).to(x.dtype)

        dout = dout.to(x.dtype)

        db = torch.zeros_like(b).to(x.dtype)
        dw = torch.zeros_like(w).to(x.device).to(x.dtype)
        dx_padding = torch.zeros_like(x_padding).to(x.device).to(x.dtype)

        for batch in range(N):
            for filters in range(F):
                db[filters] += torch.sum(dout[batch, filters])
                for height in range(0, H_out):
                    for width in range(0, W_out):
                        dw[filters] += dout[batch, filters, height, width] * x_padding[batch, :, height * stride : height * stride + HH, width * stride : width * stride + WW]
                        dx_padding[batch, :, height * stride : height * stride + HH, width * stride : width * stride + WW] += w[filters] * dout[batch, filters, height, width]

        dx = dx_padding[:, :, pad : pad + H, pad : pad + W].to(x.dtype)

        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code

        N, C, H, W = x.shape
        pool_height = pool_param.get('pool_height', 2)
        pool_width = pool_param.get('pool_width', 2)
        stride = pool_param.get('stride', 2)

        H_out = 1 + (H - pool_height) // stride
        W_out = 1 + (W - pool_width) // stride
        
        out = torch.zeros(N, C, H_out, W_out).to(x.device).to(x.dtype)

        for row in range(H_out):
          for col in range(W_out):
            col_start = col * stride
            col_end = col_start + pool_width
            row_start = row * stride
            row_end = row_start + pool_height

            out[:, :, row, col] = torch.max(torch.max(x[:, :, row_start : row_end, col_start : col_end], -1)[0], -1)[0]
        
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        
        x, pool_param = cache
        N, C, H, W = x.shape
        pool_height = pool_param.get('pool_height', 2)
        pool_width = pool_param.get('pool_width', 2)
        stride = pool_param.get('stride', 2)
        _, _, H_out, W_out = dout.shape

        dx = torch.zeros_like(x)

        for img in range(N):
          for channel in range(C):
            for row in range(H_out):
              for col in range(W_out):
                row_start = row * stride
                row_end = row_start + pool_height
                col_start = col * stride
                col_end = col_start + pool_width

                pool = x[img, channel, row_start : row_end, col_start : col_end]
                
                max_index = torch.argmax(pool)

                col_idx = max_index % pool_width
                row_idx = max_index // pool_width

                dx[img, channel, row_start:row_end, col_start:col_end][row_idx, col_idx] = dout[img, channel, row, col]


        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code

        C, H, W = input_dims
        self.params['W1'] = torch.randn(num_filters, C, filter_size, filter_size).to(dtype).to(device) * weight_scale
        self.params['b1'] = torch.zeros(num_filters).to(dtype).to(device)
        self.params['W2'] = torch.randn(num_filters * H * W // 4, hidden_dim).to(dtype).to(device) * weight_scale
        self.params['b2'] = torch.zeros(hidden_dim).to(dtype).to(device)
        self.params['W3'] = torch.randn(hidden_dim, num_classes).to(dtype).to(device) * weight_scale
        self.params['b3'] = torch.zeros(num_classes).to(dtype).to(device)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        # Replace "pass" statement with your code
        
        # WANT: [conv -> relu -> 2x2 max pool] -> [linear -> relu] -> [linear -> softmax]
        # Also have:
        #   - Conv_ReLU
        #     - forward(x, w, b, conv_param) --> return out, cache
        #     - backward(dout, cache) --> return dx, dw, db
        #   - Conv_ReLU_Pool
        #     - forward(x, w, b, conv_param, pool_param) --> return out, cache
        #     - backward(dout, cache) --> return dx, dw, db

        out1, cache1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        out2, cache2 = Linear_ReLU.forward(out1, W2, b2)
        scores, cache3 = Linear.forward(out2, W3, b3)

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # Replace "pass" statement with your code
        
        # Softmax -> Regularize -> Backpropogate -> Regularize
        soft_loss, soft_gradient = softmax_loss(scores, y)

        reg_loss = (torch.sum(W1 ** 2) + torch.sum(W2 ** 2) + torch.sum(W3 ** 2)) * self.reg
        loss = soft_loss + reg_loss

        dout2, dW3, db3 = Linear.backward(soft_gradient, cache3)
        dout1, dW2, db2 = Linear_ReLU.backward(dout2, cache2)
        dX, dW1, db1 = Conv_ReLU_Pool.backward(dout1, cache1)
        
        dW3 += self.reg * W3
        dW2 += self.reg * W2
        dW1 += self.reg * W1

        grads['W3'] = dW3
        grads['b3'] = db3
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1

        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        # Replace "pass" statement with your code

        C, H, W = input_dims
        down_sample = 2 ** len(max_pools)

        # L1 initialization
        if weight_scale == "kaiming":
          self.params['W1'] = kaiming_initializer(num_filters[0], C, 3, relu=True, dtype = self.dtype, device = device)
        else:
          self.params['W1'] = weight_scale * torch.randn(num_filters[0], C, 3, 3, dtype = self.dtype, device = device)
        self.params['b1'] = torch.zeros(num_filters[0], dtype = self.dtype, device = device)

        # L2 - LN-1 initialization
        for i in range(2, self.num_layers):
          if weight_scale == "kaiming":
            self.params['W' + str(i)] = kaiming_initializer(num_filters[i - 1], num_filters[i - 2], K=3, relu=True, dtype = self.dtype, device = device)
          else:
            self.params['W' + str(i)] = weight_scale * torch.randn(num_filters[i - 1], num_filters[i - 2], 3, 3, dtype = self.dtype, device = device)
          self.params['b' + str(i)] = torch.zeros(num_filters[i - 1], dtype = self.dtype, device = device)

        # LN initialization
        if weight_scale == "kaiming":
          self.params['W' + str(self.num_layers)] = kaiming_initializer(num_filters[-1] * H * W // down_sample ** 2, num_classes, relu=True, dtype = self.dtype, device = device)
        else:
          self.params['W' + str(self.num_layers)] = weight_scale * torch.randn(num_filters[-1] * H * W // down_sample ** 2, num_classes, dtype = self.dtype, device = device)
        self.params['b' + str(self.num_layers)] = torch.zeros(num_classes, dtype = self.dtype, device = device)

        # Batch normalization
        if self.batchnorm:
          for i in range(1, self.num_layers):
              gamma_shape = (num_filters[i - 1],)
              beta_shape = (num_filters[i - 1],)
              self.params['gamma' + str(i)] = torch.ones(gamma_shape, dtype=self.dtype, device=device).view(-1)
              self.params['beta' + str(i)] = torch.zeros(beta_shape, dtype=self.dtype, device=device).view(-1)

        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code

        cache = []
        scores = X

        # Forward: {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear
        for i in range(1, self.num_layers + 1):
          W_i = self.params['W' + str(i)]
          b_i = self.params['b' + str(i)]

          # LAST: Linear -> Predicted Classifications
          if i == self.num_layers:
            scores, c = Linear.forward(scores, W_i, b_i)
        
          else:
            # max_pool is 0 indexed
            if (i - 1) in self.max_pools:
              # Pooling layer
              if (self.batchnorm and i != self.num_layers):
                # Conv -> Batchnorm -> ReLU -> Pool
                scores, c = Conv_BatchNorm_ReLU_Pool.forward(scores, W_i, b_i, self.params['gamma' + str(i)], self.params['beta' + str(i)], conv_param, self.bn_params[i - 1], pool_param)
              else:
                # Conv -> ReLU -> Pool
                scores, c = Conv_ReLU_Pool.forward(scores, W_i, b_i, conv_param, pool_param)
            else:
              # NON-pooling layer
              if (self.batchnorm and i != self.num_layers):
                # Conv -> Batchnorm -> ReLU
                scores, c = Conv_BatchNorm_ReLU.forward(scores, W_i, b_i, self.params['gamma' + str(i)], self.params['beta' + str(i)], conv_param, self.bn_params[i - 1])
              else:
                # Conv -> ReLU
                scores, c = Conv_ReLU.forward(scores, W_i, b_i, conv_param)
          
          # Add to cache
          cache.append(c)

        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code

        loss, dout = softmax_loss(scores, y)
        
        # Indexing of Layers
        for i in range(self.num_layers, 0, -1): 
          
          # FIRST: HANDLE LAST LINEAR CLASSIFIER BACKWARDS
          if i == self.num_layers:
            dout, grads['W' + str(i)], grads['b' + str(i)] = Linear.backward(dout, cache.pop())

          else:
            # max_pool is 0 indexed
            if (i - 1) in self.max_pools:
              # Pooling layer
              if self.batchnorm:
                dout, grads['W' + str(i)], grads['b' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = Conv_BatchNorm_ReLU_Pool.backward(dout, cache.pop())
              else:
                dout, grads['W' + str(i)], grads['b' + str(i)] = Conv_ReLU_Pool.backward(dout, cache.pop())
            else:
              # NON-pooling layer
              if self.batchnorm:
                dout, grads['W' + str(i)], grads['b' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = Conv_BatchNorm_ReLU.backward(dout, cache.pop())
              else:
                dout, grads['W' + str(i)], grads['b' + str(i)] = Conv_ReLU.backward(dout, cache.pop())

          # Regularize Gradients
          grads['W' + str(i)] += self.reg * self.params['W' + str(i)] * 2

          # Compound Loss
          loss += self.reg * torch.sum(self.params['W' + str(i)] ** 2)


        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    weight_scale = 25e-2
    learning_rate = 7e-4
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code

    model = DeepConvNet(
      input_dims = data_dict["X_train"].shape[1:], 
      num_classes = 10,
      num_filters = [64, 64, 128, 128, 256, 256], 
      max_pools=[1, 2, 3],
      weight_scale = "kaiming",
      reg = 5e-4,
      dtype = dtype,
      device = device
      )

    solver = Solver(
      model, 
      data_dict,
      num_epochs = 5, 
      batch_size = 128,
      update_rule = adam,
      optim_config = {'learning_rate': 1e-3},
      lr_decay = 0.95,
      print_every = 100, 
      device = device
    )

    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        
        weight = torch.randn(Din, Dout, dtype = dtype, device = device) * ((gain / Din) ** (0.5))

        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        
        weight = torch.randn(Din, Dout, K, K, dtype = dtype, device = device) * ((gain / (Din * K * K)) ** (0.5))
        
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code

            # in_mean = torch.mean(x, dim = 0)
            # in_var = torch.var(x, dim = 0, unbiased = False)

            # normal_x = (x - in_mean) / (torch.sqrt(in_var + eps))
            # out = normal_x * gamma + beta

            # running_mean = running_mean * momentum + (in_mean * (1 - momentum))

            # running_var = running_var * momentum + (in_var * (1 - momentum))

            # cache = (x, normal_x, gamma, beta, in_mean, in_var, eps)
            
            # Compute batch mean and variance
            mean_b = 1 / N * x. sum (dim=0)
            var_b = 1 / N * (x - mean_b) .pow (2). sum (dim=0)
            
            # Update running mean and variance
            running_mean = momentum * running_mean + (1 - momentum) * mean_b
            running_var = momentum * running_var + (1 - momentum) * var_b
            
            # Batch normalization
            x_normed = (x - mean_b) / torch.sqrt(var_b + eps)
            out = gamma * x_normed + beta
            # Cache computations for backwards pass
            cache = (x, mean_b, var_b, gamma, beta, eps)

            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code
            
            # normal_x = (x - running_mean) / (torch.sqrt(running_var + eps))
            # out = gamma * normal_x + beta
            
            x_normed = (x - running_mean) / torch.sqrt(running_var + eps)
            out = gamma * x_normed + beta
            # Cache computations for backwards pass
            cache = (x, running_mean, running_var, gamma, beta, eps)

            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your code

        # x, normal_x, gamma, beta, in_mean, in_var, eps = cache
        # N, D = dout.shape
        # dbeta = torch.sum(dout, dim = 0)
        # dgamma = torch.sum(dout * normal_x, dim = 0)

        # dvar = torch.sum((dout * gamma) * (x-in_mean), 0) * (-0.5) * (eps + in_var) ** (-1.5)
        # dmean = 2 * (x-in_mean) * ((1 / N) * dvar * torch.ones(dout.shape, device = dout.device))
        
        # dx = dmean + (dout * gamma) * 1 / (in_var + eps) ** (0.5) + 1 / N * torch.ones(dout.shape).to(dout.device) * ((-1) * torch.sum((dout * gamma) * 1 / (in_var + eps) ** (0.5) + dmean, 0))
        
        # Constants
        N, D = dout.shape
        x, mean, var, gamma, beta, eps = cache
        # Calculating dL/dx hat
        dx_hat = dout * gamma
        # Calculating dL/dvar
        dvar = (dx_hat * (x - mean) ).sum(dim=0) * (-1 / ( 2 * (var + eps).pow(3).sqrt()) )
        # Calculating dL/dmean
        dmean = (dx_hat * (-1 / (var + eps).sqrt())).sum(dim=0)
        # Calculating dL/dx
        dx = ((dx_hat / (var + eps).sqrt()) + (dvar * 2 * (x - mean) / N) + (dmean / N))
        # Calculating dL/gamma
        x_normed = (x - mean) / torch.sqrt (var + eps)
        dgamma = (dout * x_normed).sum(dim=0)
        # Calculating dL/beta
        dbeta = dout.sum(dim=0)

        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta



class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code

        N, C, H, W = x.shape

        x_chan = x.permute(0, 2, 3, 1).reshape(N * H * W, C)

        out, cache = BatchNorm.forward(x_chan, gamma, beta, bn_param)

        out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)

        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code

        N, C, H, W = dout.shape

        dout_shaped = dout.permute(0, 2, 3, 1).reshape(N * H * W, C)

        dx, dgamma, dbeta = BatchNorm.backward(dout_shaped, cache)

        dx = dx.reshape(N, H, W, C).permute(0, 3, 1, 2)

        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta