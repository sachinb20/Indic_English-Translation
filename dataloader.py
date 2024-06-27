from torch.utils.data import Dataset, IterableDataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List

class MyIterableDataset(IterableDataset):
    def __init__(self, english_sentences, hindi_sentences):
        self.english_sentences = english_sentences
        self.hindi_sentences = hindi_sentences
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.english_sentences):
            raise StopIteration
        else:
            english_sentence = self.english_sentences[self.index]
            hindi_sentence = self.hindi_sentences[self.index]
            self.index += 1
            return hindi_sentence, english_sentence
        

#!git clone "https://github.com/anoopkunchukuttan/indic_nlp_library"
#!git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
#!pip install Morfessor
# The path to the local git repo for Indic NLP library
def tokenizer(SRC_LANGUAGE,TGT_LANGUAGE,train):
    INDIC_NLP_LIB_HOME=r"indic_nlp_library"

    # The path to the local git repo for Indic NLP Resources
    INDIC_NLP_RESOURCES="indic_nlp_resources"

    import sys
    sys.path.append(r'{}'.format(INDIC_NLP_LIB_HOME))
    from indicnlp import common
    common.set_resources_path(INDIC_NLP_RESOURCES)
    from indicnlp import loader
    loader.load()



    import spacy
    # !python -m spacy download en_core_web_sm
    eng = spacy.load("en_core_web_sm")

    from indicnlp.tokenize import indic_tokenize

    def engTokenize(text):
        """
        Tokenize an English text and return a list of tokens
        """
        return [str(token.text) for token in eng.tokenizer(str(text))]

    def hiTokenize(text):
        """
        Tokenize a German text and return a list of tokens
        """
        return [str(t) for t in indic_tokenize.trivial_tokenize(str(text))]

    # def getTokens(data_iter, place):
    #     """
    #     Function to yield tokens from an iterator. Since, our iterator contains
    #     tuple of sentences (source and target), `place` parameters defines for which
    #     index to return the tokens for. `place=0` for source and `place=1` for target
    #     """
    #     for english, german in data_iter:
    #         if place == 0:
    #             yield engTokenize(english)
    #         else:
    #             yield hiTokenize(german)

    # Place-holders
    token_transform = {}
    vocab_transform = {}

    token_transform[SRC_LANGUAGE] = hiTokenize
    token_transform[TGT_LANGUAGE] = engTokenize

    # function to generate the tokens for each language
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
        language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield token_transform[language](data_sample[language_index[language]])

    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        #create the iterator object of the dataset given
        train_iter = MyIterableDataset(train['english'], train['hindi'])
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=2,
                                                        specials=special_symbols,
                                                        special_first=True,
                                                        # max_tokens = vocab_size[ln]
                                                        )

    #setting the default index to unknown index which means that it will assume the token to be unknown if
    #it sees a word not in the dictionary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)



    print("tokenized hindi sentence ", token_transform['en'](train['english'][0]))
    print("numericalized hindi sentence ", vocab_transform['en'](token_transform['hi'](train['english'][0])))

    return token_transform,vocab_transform




