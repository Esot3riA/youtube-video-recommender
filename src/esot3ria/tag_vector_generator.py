import nltk
import gensim
import pandas as pd

# Load files.
nltk.download('stopwords')
vocab = pd.read_csv('../vocabulary.csv')

# Lower corpus and Remove () from name.
vocab['WikiDescription'] = vocab['WikiDescription'].str.lower().str.replace('[^a-zA-Z0-9]', ' ')
for i in range(vocab['Name'].__len__()):
    name = vocab['Name'][i]
    if isinstance(name, str) and name.find(" (") != -1:
        vocab['Name'][i] = name[:name.find(" (")]
vocab['Name'] = vocab['Name'].str.lower()

# Combine separated names.(mobile phone -> mobile-phone)
for name in vocab['Name']:
    if isinstance(name, str) and name.find(" ") != -1:
        combined_name = name.replace(" ", "-")
        for i in range(vocab['WikiDescription'].__len__()):
            if isinstance(vocab['WikiDescription'][i], str):
                vocab['WikiDescription'][i] = vocab['WikiDescription'][i].replace(name, combined_name)


# Remove stopwords from corpus.
stop_re = '\\b'+'\\b|\\b'.join(nltk.corpus.stopwords.words('english'))+'\\b'
vocab['WikiDescription'] = vocab['WikiDescription'].str.replace(stop_re, '')
vocab['WikiDescription'] = vocab['WikiDescription'].str.split()

# Tokenize corpus.
tokenlist = [x for x in vocab['WikiDescription'] if str(x) != 'nan']
phrases = gensim.models.phrases.Phrases(tokenlist)
phraser = gensim.models.phrases.Phraser(phrases)
vocab_phrased = phraser[tokenlist]

# Vectorize tags.
w2v = gensim.models.word2vec.Word2Vec(sentences=tokenlist, min_count=1)
w2v.save('tag_vectors.model')

# word_vectors = w2v.wv
# vocabs = word_vectors.vocab.keys()
# word_vectors_list = [word_vectors[v] for v in vocabs]
