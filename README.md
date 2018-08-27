### Project Flecter Summary

#### Project Design

The main focus on my project was to build a twitter bot that would capture the linguistic nuiance of the crypto community /meme space.
I chose this community as the community has a specific lingo that is interesting and highly entertaining. I also wanted to explore language generation
and the so called "unreasonable effectiveness of RNNs"

#### Data

I chose to take data from the reddit community using the PRAW api, which only allows you to query up to a 1000 comments per subreddit. In this instance I managed
to capture quite a lot of data, howeve I ended up with a second interation after finding a website which lists over 800 crypto themed reddits. In this final 
iteration I was able to capture over 750k sentances out of the 84k comments (after standardizes each sentance to a fixed length of 100 words.

#### TOOLS

```python3
praw
nltk 
gensim
pytorch
sklearn
```

computing: AWS

#### DATA PROCESSING
Since the "language" of this community isn't completely english and contains made up words, I did not use english stop words. I instead cleaned up
the text to remove excess spaces, links, and puntuation that wasn't [,.!?]

#### WORD2VECT
I explored the space with word2vect, to help me understand what is was some of these "foreign" langauge words could also mean. It did a very 
accurate job on categorizing things like bitcoin, the ethereum creator (Vitalik Buterin) and also characterizing the word "shiba" as in doge coin
memes. The idea was to use this in the LSTM, however I ended up using a one hot encoding on all the words

#### TEXT GENERATION - MARKOV APPROACH
To start out with tweet generation, I used markov chain with a bigram. Much like the pair programming problem , the output ended up being
poetic and believable grammatical tweets. However, since the markov chain does not abstract context, it doesn't quite sound like something
that would actually make sense when talking about the topics "bitcoin transaction, good question?" for example, is not really a question 
that needs to be addressed. However, I got some real gems such as "shiba inu memes rule". Note, I seeded the markov chain with real
words that are specific to this community. 

### TEXT GENERATION - LSTM APPROACH
I used pytorch and decided to use an LSMT with a batch size of 32. This was done by first one hot encoding my sentances.
I ended up using at first 1layer, however this was giving me sentances that were, although some what coherent, they didn't quite make
sense. I decided to go up to 3layers. This actually gave me some tweets that associated the topic with things about that topic
that are unique. For instance, ethereum got the creator (vitalik) and also web3, which is a decentralized internet built on the ethereum 
block chain. Pretty cool!

Finally, I chose to explore the character by character LSMT. This was super fast and gave me very clean sentances in the end for a given temperature.
I noticed I couldn't lower the temperature too much or it would converge on some of the most talked about topics (platforms and projects).
If the temperature was too high I got just a bunch of links and crap that didn't make sense. I decided to go with a temperature of 0.55. While topics
and platforms still made an appeareance, other topics were able to come through as well. 
  

#### CLUSTERING 
Since I already had my dimensionality reduced using word2vect, I took a random sampling of my sentances and used Kmeans to cluster the topics.
I created a TSNE plot and labelled the centroid to see if this would help me find clusters. I actually believe I got a cryptocurrency topic,
but I know now I should have spent more time looking at words other than the centroid to determine the topic of cluster. The remaining data doesn't
look as promising as it doesn't cluster as well and the topics I got were words like "hate" and "no" which doesn't really help determine the topic.
I would have liked to use LDA or another dimensionality reducer to help me cluster. 

#### CHALLENGES
It was extremely hard to get the LSMT to complete even a single epoch. I wanted to go to 10, however all the tweets presented were only the
first epoch.
The reason:
I am learning the hard way that time complexity is extremely important. So is incremental check points and being able to assess in real time
whether your results are as expected. 

The word-by-word approach to the LSMT is ultimately slow, even when processing in batches. Running a gpu may have helped, however 
perhaps a better approach would be to subsample and look at the loss function to see if it can converge faster. Then a sampling of sentances
maybe be a better method than to process blocks and blocks of sentences with little return from the learning perspective. 
Also, I got unlucky and had a bad image from my AWS instance, something that I thought was my fault and spent time trying to resolve
gpu drivers and the like. In the future I will test to see if the image is bad ahead of time (load up python and import torch and test if
the cuda core can be seen)

I would have explored the space up front with clustering and then used the topics to generate tweets, 
especially if I were to reuse this technique for learning the language of a community I don't know much about ahead of time. 

#### FUTURE WORK
Word2Vect was explored, however it would be better to see if it could reduce the work of the LSMT. Additionally, I would like 
to get this to run on a gpu to test to see how much time it actually saves. Additionally, I would like to monitor my loss function
with an external python package that let's you see graphs and variables in real time.

