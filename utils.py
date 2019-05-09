
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
from datetime import datetime
import codecs
import collections

# .. to-do::
#
#   - add <beg> and <end> tokens
#   - potentially add <pad> token
#   - test the dataset object!
#       - create method for generating large 'torchified', padded, tokenized, one-hotted torch tensor
#           that gets sent into PrcDataSet

def load_tweets(f, after_date=None, silent=True, encoding='utf-8'):
    """
        Load tweets from given csv file, using the header information
        to determine what column the tweet is contained in.

        .. note:: 
            
            - retweets aren't included
            - I think I may need to make an end and beggining of tweet token
    """
    tweet_date_fmt = '%m-%d-%Y %H:%M:%S'

    # get raw data
    with codecs.open(f,'r',encoding=encoding) as fl:
        csv_rdr = csv.reader(fl)
        data = [ row for row in csv_rdr ]

    # get header and get indices for useful fields
    header      = data.pop(0)
    tweet_ind   = header.index('text')
    date_ind    = header.index('created_at')
    retweet_ind = header.index('is_retweet')

    # get relevant tweets (sans retweets)
    tweets = []
    for row in data:
        try:
            if row[retweet_ind] == 'false':
                if after_date:
                    # skip tweet if it was before given date
                    tweet_date = datetime.strptime(row[date_ind], tweet_date_fmt)
                    if tweet_date <= after_date: continue
                tweets.append(row[tweet_ind])
        except IndexError:
            if not silent: 
                print("Index Error occured. Skipping row")
                print("Row: {}".format(",".join(row)))
            continue
    return tweets

def get_vocab(tweets,bot='<BOT>',eot='<EOT>'):
    """
        Get list of observed characters in given tweet list.
    """
    # get counter object of characters to determine what chars are used
    counter     = collections.Counter(" ".join(tweets))
    count_pairs = sorted(counter.items(), key=lambda x: -x[1]) # sort by counts
    vocab, _    = zip(*count_pairs)

    # add beginning of tweet and end of tweet tokens to vocab
    vocab = (bot,eot) + vocab
    return vocab

def make_dicts(vocab):
    """
        Given a vocabularly, create dictionaries to map between characters and 
        integers.
    """
    int2char = dict(enumerate(vocab))
    char2int = { ch : i for i,ch in int2char.items() }
    return int2char, char2int

def tokenize_and_onehot_encode(tweet, char2int, bot='<BOT>', eot='<EOT>'):
    """
        Given a tweet, and a specified char2int dictionary, create
        a tokenized and one hot encoded matrix, representing the given 
        tweet.
    """
    # tokenize raw tweet
    tkn_tweet = np.array([char2int[char] for char in tweet])

    # add beginning of tweet and end of tweet tokens
    tkn_tweet = np.insert(tkn_tweet,0,char2int[bot])
    tkn_tweet = np.append(tkn_tweet,char2int[eot])

    # set vocab size and initialize one hot tokenized tweet
    num_chars       = len(char2int)
    oh_tkn_tweet    = np.zeros(shape=(len(tkn_tweet),num_chars), dtype=np.float32)

    # set proper values to 1
    oh_tkn_tweet[np.arange(len(tkn_tweet)),tkn_tweet] = 1.
    return oh_tkn_tweet

def create_store_tensors(tweets, char2int, bot='<BOT>', eot='<EOT>',
                            pad_value=np.nan, data_dir='data'):
    """
        Create and store tokenized, one hot vector, pytorch tensors in a data directory.

        To use this to create a dataset, run:
            from utils import *
            from datetime import datetime
            tweet_f = 'tweet_datasets/TrumpTwitterArchive/20090504_20190422.csv'
            bot,eot = '<BOT>','<EOT>'
            after_date = datetime(2015,1,1,0)
            tweets = load_tweets(tweet_f,after_date=after_date)
            vocab = get_vocab(tweets,bot,eot)
            int2char,char2int = make_dicts(vocab)
            IDs = create_store_tensors(tweets,char2int,bot,eot)
            dataset = PrcDataSet(IDs)

        Parameters
        ----------
            tweets : list of str
                list of raw tweets
            char2int : dict
                dictionary mapping between characters and integers
            bot : str **optional**
                beginning of tweet token. Defaults to '<BOT>'
            eot : str **optional**
                end of tweet token. Defaults to '<EOT>'.
            pad_value : float **optional**
                the value that will be used to pad entries in tweets
                to make the lengths equal. Defaults to ``numpy.nan``
            data_dir : str **optional**
                to directory where the .pt files will be stored

        Returns
        -------
            IDs : list of tuples containing integers
                IDs for all saved tensors and the true length of each tweet    
    """

    # get number of characters
    num_chars = len(char2int)

    # tokenize and onehot encode tweets and store in list
    #   note: tokenize_and_onehot_encode adds bot and eot to the tweet!
    tkn_tweets_lst = []
    for tweet in tweets:
        tkn_tweet = tokenize_and_onehot_encode(tweet,char2int,bot=bot,eot=eot)
        tkn_tweets_lst.append(tkn_tweet)

    # sort list so that max length tweet comes first
    tkn_tweets_lst = sorted(tkn_tweets_lst, key=len, reverse=True)
    max_len = len(tkn_tweets_lst[0])

    # torch tensor from each tokenized tweet and save in the datadir
    if not os.path.isdir(data_dir): os.mkdir(data_dir)
    IDs = []
    for i,tweet in enumerate(tkn_tweets_lst):
        if not i == 0:
            # pad tweet before adding to array
            tweet_len = len(tweet)
            pad_len = max_len - tweet_len
            pad_arr = np.full(shape=(pad_len,num_chars),fill_value=pad_value)
            pad_tweet = np.append(tweet,pad_arr,axis=0)
        else: 
            # padding isn't required
            tweet_len = max_len
            pad_tweet = tweet
        
        # save tensor
        tnsr_f = os.path.join(data_dir,'{}.pt'.format(i))
        torch.save(torch.from_numpy(tweet),tnsr_f)
        IDs.append((i,tweet_len))

    print("Processed tweets save in tokenized, one-hot torch tensors in {}".format(data_dir))
    return IDs

class PrcDataSet(Dataset):
    """ Processed tweet dataset
    """

    def __init__(self, IDs, data_dir='data'):
        """ initiate data set with large torch tensor

            Parameters
            ----------
                IDs : list of ints
                    defines the IDs of the torch tensor files 
        """
        self.IDs        = IDs
        self.data_dir   = data_dir
    
    def __len__(self):
        return len(self.IDs)

    def __getitem__(self,idx):
        """
        Method for getting the ith sequence and labels, and the sequence length

        .. note::

            if ``seq`` is the whole sequence, then X = seq[:-1], and lbls = seq[1:]

        Returns
        -------
            X : torch tensor with dtype = torch.int8
                padded/tokenized tweet sequence in one-hot vector format. 
                Dim: ( lngth x num_chars )
            lbls : torch tensor with dtype = torch.int8
                padded/tokenized tweet sequence in one-hot vector format
                Dim: ( lngth x num_chars )
            lngth : torch.int8
                length of the sequence X
        """
        ID      = self.IDs[idx][0]
        lngth   = self.IDs[idx][1]

        # load tensor
        tensor_f= os.path.join(self.data_dir,'{}.pt'.format(idx))
        tensor  = torch.load(tensor_f)
        X       = tensor[:-1,:]
        lbls    = tensor[1:,:]
        return X,lbls,lngth
