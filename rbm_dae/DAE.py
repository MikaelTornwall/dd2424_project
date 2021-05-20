import torch
import torch.nn as nn

class DAE(nn.Module):
    """A Deep Autoencoder that takes a list of RBMs as input"""

    def __init__(self, models,use_models = True):
        """Create a deep autoencoder based on a list of RBM models
        Parameters
        ----------
        models: list[RBM]
            a list of RBM models to use for autoencoding
        """
        super(DAE, self).__init__()

        # extract weights from each model
        encoders = []
        encoder_biases = []
        decoders = []
        decoder_biases = []
        for model in models:
            encoders.append(nn.Parameter(model.W.clone()))
            encoder_biases.append(nn.Parameter(model.h_bias.clone()))
            decoders.append(nn.Parameter(model.W.clone()))
            decoder_biases.append(nn.Parameter(model.v_bias.clone()))

        # build encoders and decoders
        self.encoders = nn.ParameterList(encoders)
        self.encoder_biases = nn.ParameterList(encoder_biases)
        self.decoders = nn.ParameterList(reversed(decoders))
        self.decoder_biases = nn.ParameterList(reversed(decoder_biases))

        if not use_models:
            for encoder in self.encoders:
                torch.nn.init.xavier_normal_(encoder, gain = 1.0)
            for encoder_bias in self.encoder_biases:
                torch.nn.init.zeros_(encoder_bias)
            for decoder in self.decoders:
                torch.nn.init.xavier_normal_(decoder, gain = 1.0)
            for decoder_bias in decoder_biases:
                torch.nn.init.zeros_(decoder_bias)


    def forward(self, v):
        """Forward step
        Parameters
        ----------
        v: Tensor
            input tensor
        Returns
        -------
        Tensor
            a reconstruction of v from the autoencoder
        """
        # encode
        p_h = self.encode(v)

        # decode
        p_v = self.decode(p_h)

        return p_v

    def encode(self, v):  # for visualization, encode without sigmoid
        """Encode input
        Parameters
        ----------
        v: Tensor
            visible input tensor
        Returns
        -------
        Tensor
            the activations of the last layer
        """
        p_v = v
        activation = v
        for i in range(len(self.encoders)):
            W = self.encoders[i]
            h_bias = self.encoder_biases[i]
            activation = torch.mm(p_v, W) + h_bias
            p_v = torch.sigmoid(activation)

        # for the last layer, we want to return the activation directly rather than the sigmoid
        return activation

    def decode(self, h):
        """Encode hidden layer
        Parameters
        ----------
        h: Tensor
            activations from last hidden layer
        Returns
        -------
        Tensor
            reconstruction of original input based on h
        """
        p_h = h
        for i in range(len(self.encoders)):
            W = self.decoders[i]
            v_bias = self.decoder_biases[i]
            activation = torch.mm(p_h, W.t()) + v_bias
            p_h = torch.sigmoid(activation)
        return p_h