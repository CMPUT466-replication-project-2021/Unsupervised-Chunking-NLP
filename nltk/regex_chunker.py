import nltk
from nltk.corpus import conll2000
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

grammar = "NP: {(<DT><NN>)|<PRP>|<NNS>|<NNP>|<NN>|(<NNP><NNP>)|(<DT><JJ><NN>)|(<JJ><NNS>)|(<DT><NNS>)|(<JJ><NN>)}"
regexp_chunker = nltk.RegexpParser(grammar)
print(regexp_chunker.evaluate(test_sents))

grammar = r"NP: {<[CDJNP].*>+}"
regexp_chunker = nltk.RegexpParser(grammar)
print(regexp_chunker.evaluate(test_sents))