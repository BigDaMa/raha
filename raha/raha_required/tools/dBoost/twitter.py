#! /usr/bin/env python3

from TwitterSearch import *
from credentials import *
from config import *
import json
from dateutil import parser
import datetime
import os

def date_to_string(date):
    return datetime.datetime.strftime(date, '%a %b %d %H:%M:%S %z %Y')

def load_tweet_times():
    if os.path.isfile(time_limits_file): 
        with open(time_limits_file, 'r') as file:
            ltt = json.loads(file.read())
            ltt = {key:parser.parse(ltt[key]) for key in ltt}
    else:
        ltt = {}

    for user in users:
        if user not in ltt:
            ltt[user] = parser.parse(default_start_time)

    return ltt

def save_tweet_times(ltt):
    with open(time_limits_file, "w") as file:
        ltt = {key:date_to_string(ltt[key]) for key in ltt}
        file.write(json.dumps(ltt))

def authenticate():
    return TwitterSearch(
        consumer_key = CONSUMER_KEY,
        consumer_secret = CONSUMER_SECRET,
        access_token = ACCESS_TOKEN,
        access_token_secret = ACCESS_TOKEN_SECRET
    )

time_limit = load_tweet_times()
new_time_limit = time_limit.copy()

with open(output_file, "a") as file:
    try:
        for user in users:
            tuo = TwitterUserOrder(user)
            ts = authenticate()
            obtained_tweets = 0

            for tweet in ts.search_tweets_iterable(tuo):
                cur_time = parser.parse(tweet['created_at'])

                if cur_time > new_time_limit[user]: # after finishing, we will have tweets up to this point
                    new_time_limit[user] = cur_time
                        
                if cur_time <= time_limit[user]: # we already have these tweets
                    print ("Got {} tweet(s) from {}".format(obtained_tweets, user))
                    break;
    
                obtained_tweets += 1
                file.write(json.dumps(tweet))
                file.write('\n')
    
    except TwitterSearchException as e: 
        print(e)
                
    finally:
        save_tweet_times(new_time_limit)
        file.close()
