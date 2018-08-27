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
