import numpy as np

# TODO: Document this absolute mess.

class BatchGenerator(object):
    def __init__(self, preprocessed_data, batch_size=12, approx_n_buckets = 5):
        self.preprocessed_data = preprocessed_data
        self.data = self.preprocessed_data.get_data()
        self.batch_size = batch_size
        self.approx_n_buckets = approx_n_buckets
        self._seq_length = [(len(joke),joke) for joke in self.data]
        self.lengths = np.unique([len(joke) for joke in self.data])
        self._seq_length.sort(key=lambda x: x[0])
        self.bucket_size = len(self.data) // (self.approx_n_buckets)
        self.buckets = list()
        self.grouptuple = list()
        self._group()
    
    def nextBatch(self):
        bucket_sample = np.random.randint(0,len(self.buckets),1)[0]
        bucket = self.buckets[bucket_sample]
        batch = np.zeros((self.batch_size, len(bucket[0]), self.preprocessed_data.vocab_size))
        seq_length = len(bucket[0])
        for i in range(self.batch_size):
            random_sample = np.random.randint(0,len(bucket),1)[0]
            for k,j in enumerate(bucket[random_sample]):
                batch[i, k, j] = 1
        return seq_length, batch
            
    
    def _group(self):
        counts = np.zeros((len(self.lengths)))
        length2ind = dict([length,i] for i,length in enumerate(self.lengths))
        ind2length = dict([i,length] for i,length in enumerate(self.lengths))
        for seq_length, value in self._seq_length:
            counts[length2ind[seq_length]] += 1
        groups = list()
        n_elements = 0 
        groups.append(n_elements)
        for i in range(len(counts)):
            n_elements += counts[i]
            if n_elements > self.bucket_size:
                groups.append(i)
                n_elements = 0
        groups.append(len(counts))
        for i in range(1,len(groups)):
            self.grouptuple.append((ind2length[groups[i-1]],ind2length[groups[i]-1]))
            self.buckets.append(self._generate_bucket([ind2length[j] for j in range(groups[i-1],groups[i])]))
        assert len(self.data) == sum([len(bucket) for bucket in self.buckets])            
        
    def _generate_bucket(self, seqlengths):
        bucket = list()
        maxlen = seqlengths[-1]
        for joke in self.data:
            if len(joke) in seqlengths:
                if len(joke) < maxlen:
                    joke = self._pad(joke, maxlen)
                bucket.append(joke)
        return bucket

    def _pad(self, joke, length):
        for _ in range(len(joke),length):
            joke.append(self.preprocessed_data.w2i(self.preprocessed_data.PAD))
        assert len(joke) == length
        return joke