import spacy
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
import string

# loading data, either English or Chinese
ENGLISH = False
txt = "english music lyrics.txt" if ENGLISH else "mandarin music lyrics.txt"
with open(txt,"r",encoding = 'utf-8') as f:
    data = f.read()

# pre-processing data: remove punctuations and white spaces: chinese lyrics don't contain punctuations
# using regular expression to remove the title the songs inside<>,《》, and the english vocab in Chinese music.
# normalization: lowercase

text = re.sub('<.+>', '',data) if ENGLISH else re.sub('[A-Za-z]+|《.+》', '',data)
language ='en_core_web_lg' if ENGLISH else 'zh_core_web_lg'
nlp = spacy.load(language)
doc = nlp(text)
lyrics = [token.text
         for token in doc
         if not token.is_punct and not token.is_space]
lyrics = [w.lower() for w in lyrics]
print(f'length of lyrics:{len(lyrics)} ; preprocessed lyrics:{lyrics} ')

#baseline: random English text
def generate_random_englishword(length, alphabet=string.ascii_lowercase):
    return ''.join(random.choices(alphabet, k=length))
# the longest English word is 'pneumonoultramicroscopicsilicovolcanoconiosis', which has 45 letters,
# and 2958 is length of english lyrics
random_words = [generate_random_englishword(random.randint(2, 45)) for _ in range(2958)]
print(f'length of randomly genarated englishtext:{len(random_words)} ; randomly generated text: {random_words} ')

# divide the lyrics into noun, adjective，adverb, verb subset: open class words

pos = ["NOUN", "ADJ", "ADV", "VERB"]
pos_types = {pos_tag: [token.text for token in doc if token.pos_ == pos_tag] for pos_tag in pos}
for pos_tag, w in pos_types.items():
    print(f'number of {pos_tag}s: {len(w)}')
# count word frequency and its corresponding length, plot them.
def ZLA(type):
    word_freq = Counter(type)
    common_words = word_freq.most_common(5)
    # visualization
    sns.set_theme(style="darkgrid")
    df = pd.DataFrame.from_records(list(dict(Counter(type)).items()), columns=['word','frequency'])
    df['length'] = [len(word) for word in df['word']]
    df = df.sort_values(by=['length'], ascending=True)
    sns.relplot(x="length", y="frequency", data=df)
    plt.show()
    plt.close()
    return common_words, df

print(ZLA(pos_types['VERB']))
print(ZLA(pos_types['ADJ']))
print(ZLA(pos_types['ADV']))
print(ZLA(pos_types['NOUN']))
print(ZLA(lyrics))
print(ZLA(random_words))