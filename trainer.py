#!/usr/bin/env python
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import argparse
import os
from datetime import datetime
import json

import nets
import utils

def train(net, dataset, epochs, batch_size, lr, optm='Adam'):
    """ Take in a given network and train it with the given parameters.

        .. to-do ::

            - rewrite how the loss is calculated, using the ignore_index argument
              from NLLLoss. This would require making a specific padding token.
              This would allow us to only do one loop, over the max sequence
              length, and the loss calc would be intelligent to ignore
              contributions to the loss from padding tokens. If I get this on
              Westgrid's GPUS.. I DEFINITELY need to do this to get max
              performance.
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
            # sort batch
            srt_inps, srt_lbls, srt_lens = utils.sort_batch(inps,lbls,lens)

            # get outputs
            opt.zero_grad()
            outputs = net.forward(srt_inps,srt_lens)

            # calc loss -> sub-optimal, see trainer doc string
            batch_loss = 0
            for j in range(batch_size):

                # get length of sequence and loop over each sample
                seq_ln      = srt_lens[j]
                seq_loss    = 0
                for k in range(seq_ln):
                    opt_k = outputs[j,k,:].unsqueeze(0)
                    lbl_k = srt_lbls[j,k].unsqueeze(0).long()
                    seq_loss += loss(opt_k,lbl_k)/seq_ln
                batch_loss += seq_loss/batch_size
            print(batch_loss)

            # calc gradients, clip, and step
            batch_loss.backward()
            nrm = clip_grad_norm_(net.parameters(),0.1)
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
    parser.add_argument('-l', '--load_dataset', action='store_true',
                            help=('make the trainer load the dataset in the current data directory'+
                                  ' instead of generating it.'),
                            default=False)

    # parse arguments and set additional args
    args        = parser.parse_args()
    epochs      = args.epochs
    batch_size  = args.batch_size
    nhidden     = args.nhidden
    nlayers     = args.nlayers
    output_dir  = args.strg_dir
    tweet_file  = args.tweet_file
    bot,eot     = '<BOT>','<EOT>'       # beginning/end of tweet tokens
    after_date  = datetime(2018,1,1,0)  # date cutt off for considering tweets
    ID_f        = os.path.join('data','data_info.txt')
    char2int_f  = os.path.join('data','char2int.json')
    int2char_f  = os.path.join('data','int2char.json')

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

    # create dataset of saved, torch tensors
    if not args.load_dataset:
        print("Loading raw tweets...")
        tweets = utils.load_tweets(tweet_file,after_date=after_date)
        print("Found {} tweets".format(len(tweets)))

        # get tweet vocabulary and create mappings between integers/characters
        print("Analyzing corpus...")
        vocab  = utils.get_vocab(tweets,bot,eot)
        nchars = len(vocab) 
        int2char,char2int = utils.make_dicts(vocab)
        print("Found {} tokens".format(nchars))
        print("Generating torchified dataset...")
        IDs = utils.create_store_tensors(tweets,char2int,bot,eot)
        dataset = utils.PrcDataSet(IDs)

        # save ID file and dataset information, for later use
        with open(ID_f, 'w') as f:
            # write meta data that will be used to ID the dataset
            after_date_str = after_date.strftime("%Y-%m-%d")
            f.write('tweet_file = {}\n'.format(tweet_file))
            f.write('after_date = {}\n'.format(after_date_str))
            f.write('bot = {}\n'.format(bot))
            f.write('eot = {}\n'.format(eot))
            f.write('nchars = {}\n'.format(nchars))

            # write IDs and lengths
            f.write("\n".join("{} {}".format(tmp[0],tmp[1]) for tmp in IDs))

        with open(int2char_f,'w') as f:
            json.dump(int2char,f,indent=2,sort_keys=True)
        with open(char2int_f,'w') as f:
            json.dump(char2int,f,indent=2,sort_keys=True)
    else:
        print("Loading pre-generated processed data...")

        # char to integer and back mappings
        with open(int2char_f,'r') as f:
            int2char = json.load(f)
        with open(char2int_f,'r') as f:
            char2int = json.load(f)
        
        # load main file
        #   - I could definitely store this in a better way...
        with open(ID_f,'r') as f:
            tmp_dat = f.readlines()

        # parse meta data -> I don't like this method of loading
        tweet_file      = tmp_dat[0].split('=')[-1].strip()
        after_date_str  = tmp_dat[1].split('=')[-1].strip()
        after_date      = datetime.strptime(after_date_str,"%Y-%m-%d")
        bot             = tmp_dat[2].split('=')[-1].strip()
        eot             = tmp_dat[3].split('=')[-1].strip()
        nchars          = int(tmp_dat[4].split('=')[-1].strip())

        # IDs and sequence lengths
        tmp_dat = tmp_dat[5:]
        IDs     = [ (int(tmp.split()[0]),int(tmp.split()[1])) for tmp in tmp_dat ]

        # set dataset
        dataset = utils.PrcDataSet(IDs)

    #=========================
    # Define network and train
    #=========================
    print("Training...")
    net = nets.batch_lstm(nchars=nchars, hsize=nhidden, nlayers=nlayers)
    train(net, dataset, epochs, batch_size, lr=0.0001, optm='Adam')
