class CRFModel:
    # def __init__(self):
    #     self.
    def _word2features_pos(self, sent, i, window):
        word = sent[i][0]
        postag = sent[i][1]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        for w in range(1, window + 1):
            if i > 0:
                word = sent[i - w][0]
                postag1 = sent[i - w][1]
                features.update({
                    '-{}:word.lower()'.format(w): word.lower(),
                    '-{}:word.istitle()'.format(w): word.istitle(),
                    '-{}:word.isupper()'.format(w): word.isupper(),
                    '-{}:postag'.format(w): postag,
                    '-{}:postag[:2]'.format(w): postag[:2],
                })
            else:
                features['-{}:BOS'.format(w)] = True

            if i < len(sent) - w:
                word1 = sent[i + w][0]
                postag1 = sent[i + w][1]
                features.update({
                    '+{}:word.lower()'.format(w): word1.lower(),
                    '+{}:word.istitle()'.format(w): word1.istitle(),
                    '+{}:word.isupper()'.format(w): word1.isupper(),
                    '+{}:postag'.format(w): postag1,
                    '+{}:postag[:2]'.format(w): postag1[:2],
                })
            else:
                features['-{}:EOS'.format(w)] = True
        return features

    def _word2features(self, sent, i, window):
        word = sent[i][0]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        for w in range(1, window + 1):
            if i > 0:
                word = sent[i - w][0]
                features.update({
                    '-{}:word.lower()'.format(w): word.lower(),
                    '-{}:word.istitle()'.format(w): word.istitle(),
                    '-{}:word.isupper()'.format(w): word.isupper(),
                })
            else:
                features['-{}:BOS'.format(w)] = True

            if i < len(sent) - w:
                word1 = sent[i + w][0]
                features.update({
                    '+{}:word.lower()'.format(w): word1.lower(),
                    '+{}:word.istitle()'.format(w): word1.istitle(),
                    '+{}:word.isupper()'.format(w): word1.isupper(),
                })
            else:
                features['-{}:EOS'.format(w)] = True
        return features

    def word2features(self, sent, i, is_pos, window):
        if is_pos:
            return self._word2features_pos(sent, i, window)
        return self._word2features(sent, i, window)

    def sent2features(self, sent, is_pos, window):
        return [self.word2features(sent, i, is_pos, window) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for token, postag, label in sent]

    def sent2tokens(self, sent, is_pos):
        if is_pos:
            return [token for token, postag, label in sent]
        return [token for token, label in sent]


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                           s['POS'].values.tolist(),
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Word_idx').apply(agg_func)
        self.sentences = [s for s in self.grouped]
