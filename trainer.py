import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from datetime import datetime

import nets
import utils

def train(net, dataset, epochs, batch_size, lr, optm='Adam'):
    """ Take in a given network and train it with the given parameters.
    """
    # set into training mode
    net.train()

    # set training alg
    if optm == 'Adam':
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        raise Exception("currently only supports Adam optimizer!")

    # define loss function
    loss = nn.NLLLoss()

    # create dataloader
    data_ldr = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    # launch trainer
    for i in range(epochs):
        for inps, lbls, lens in data_ldr:
            opt.zero_grad()
            outputs     = nets.forward(inps,lens)
            batch_loss  = loss(outputs,lbls)
            print(batch_loss)
            batch_loss.backward()
            opt.step()

if __name__ == '__main__':
    
    # define CL arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs',       type=int, help='number of training epochs considered')
    parser.add_argument('batch_size',   type=int, help='batch size for each training epoch')
    parser.add_argument('nhidden',      type=int, help='hidden dimension of each LSTM layer')
    parser.add_argument('nlayers',      type=int, help='number of LSTM layers to consider')
    parser.add_argument('-s', '--strg_dir', metavar='STORAGE_DIRECTORY', action='store', 
                            default='output', help='directory to save model and loss data')
    parser.add_argument('-t', '--tweet_file', metavar='TWEET_FILE', action='store',
                            default='tweet_datasets/TrumpTwitterArchive/20090504_20190422.csv',
                            help='csv file containing the desired training data')

    # parse arguments and set additional args
    args        = parser.parse_args()
    epochs      = args.epochs
    batch_size  = args.batch_size
    nhidden     = args.nhidden
    nlayers     = args.nlayers
    output_dir  = args.strg_dir
    tweet_file  = args.tweet_file
    bot,eot     = '<BOT>','<EOT>'       # beginning/end of tweet tokens
    after_date  = datetime(2015,1,1,0)  # date cutt off for considering tweets

    print("Training a network with the given parameters")
    print("\thidden dimension size : {}".format(nhidden))
    print("\tnumber of LSTM layers : {}".format(nlayers))
    print("for {} epochs, with a batch size of {}\n".format(epochs,batch_size))

    # create output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    #====================
    # Pre-Process Dataset
    #====================
    print("Loading raw tweets...")
    tweets = utils.load_tweets(tweet_file,after_date=after_date)

    # get tweet vocabulary and create mappings between integers/characters
    print("Analyzing corpus...")
    vocab  = utils.get_vocab(tweets,bot,eot)
    nchars = len(vocab) 
    int2char,char2int = utils.make_dicts(vocab)

    # create dataset of saved, torch tensors
    print("Generating torchified dataset...")
    IDs = utils.create_store_tensors(tweets,char2int,bot,eot)
    dataset = utils.PrcDataSet(IDs)

    #=========================
    # Define network and train
    #=========================
    net = nets.batch_lstm(nchars=nchars, hsize=nhidden, nlayers=nlayers)
    train(net, dataset, epochs, batch_size, lr=0.0001, optm='Adam')
