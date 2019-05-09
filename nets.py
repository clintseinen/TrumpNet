import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class batch_lstm(nn.Module):
    def __init__(self, nchars, hsize, nlayers, dropout=0)
        """Initialize network
        """
        super(batch_lstm, self).__init__()

        # attach important variables to class
        self.nchars     = nchars
        self.hsize      = hsize
        self.nlayers    = nlayers
        self.dropout    = dropout

        # define lstm layer
        self.lstm = nn.LSTM(nchars, hsize, nlayers,
                                dropout=dropout, batch_first=True)

        # compute across token dimension
        #   - use log softmax so we can easily use negative log loss funcion (NLLoss)
        self.softmax = nn.LogSoftmax(dim=2)

        def init_hidden(self,batch_size):
            """ initialize hidden states for all layers, with dimensionality of 
                number of layers X batch size X hidden size.
            """
            h0 = torch.zeros(self.nlayers,batch_size,self.hsize)
            c0 = torch.zeros(self.nlayers,batch_size,self.hsize)
            return h0,c0

        def forward(self,inp,slens):
            """ Run the batched input sequences through the LSTM network, 
                using padded sequences.

                Parameters
                ----------
                    inp : 3D torch tensor
                        batched/padded input sequences, sorted with the longest 
                        sequence occuring first. (Batch size x Seq Len x Input size (num tokens))
                    slens : 1D torch_tensor 
                        true lengths of sequences to be padded.

                Returns
                -------
                    otp : torch tensor
                        output tensor, for each t, containing the probabilities
                        for each character token. (Batch size x Seq Len x Input size (num tokens))
            """
            # get batch size from input tensor
            batch_size = inp.shape[0]
            
            # init hidden and cell states
            h0,c0 = self.init_hidden(batch_size)

            # transform padded input tensor into "PackedSquence" object
            pckd_inp = nn.utils.rnn.pack_padded_sequence(inp,slens,batch_first=True)

            # feed through lstm, getting a 
            tmp_otp = self.lstm(pckd_inp,(h0,c0))

            # repad "PackedSequence" object
            otp = nn.utils.rnn.pad_packed_sequence(tmp_otp,padding_value=float('nan'),
                                                    batch_first=True)

            # run through softmax layer
            otp = self.softmax(otp)
            return otp
