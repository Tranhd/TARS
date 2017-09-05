from preprocess import Preprocess
from batchgenerator import BatchGenerator
import numpy as np
import progressbar
import logging
import time
import tensorflow as tf 

def logprob(predictions, labels):
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

data = Preprocess()
batch = BatchGenerator(preprocessed_data=data, batch_size=24)
steps = len(data.get_data()) // batch.batch_size
epochs = 8
vocab_size=data.vocab_size
embedding_size=212
hidden_nodes=412
keep_prob=0.5
learning_rate = 1e-3
batch_size = batch.batch_size

widgets = [
        progressbar.Percentage(),
        progressbar.Bar(),
        progressbar.DynamicMessage('loss'), ' ',
        progressbar.AdaptiveETA(),
]

bar = progressbar.ProgressBar(widgets=widgets, max_value=steps-1)

graph = tf.Graph()
with graph.as_default():

	# Input placeholders 
	with tf.variable_scope('input'):
		seqlen = tf.placeholder(tf.int32, [None])
		x = tf.placeholder(tf.float32, [None, None, vocab_size])

	# Embedding layer 
	with tf.variable_scope('embedding'):
		embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
		embedding_output = tf.nn.embedding_lookup(embeddings, tf.argmax(x, axis=2)) 

	# Cell definition
	with tf.variable_scope('LSTM_cell'):
		cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_nodes, state_is_tuple=True)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

	# RNN
	with tf.variable_scope('LSTM'):
		outputs, states  = tf.nn.bidirectional_dynamic_rnn(
		                                    cell_fw=cell,
		                                    cell_bw=cell,
		                                    dtype=tf.float32,
		                                    sequence_length=seqlen,
		                                    inputs=embedding_output)
		output_fw, output_bw = outputs
		states_fw, states_bw = states
		output = tf.reshape(output_fw, [-1, hidden_nodes])

	# Logits
	with tf.variable_scope('logits'):
		w = tf.Variable(tf.truncated_normal([hidden_nodes, vocab_size], -0.5, 0.5))
		b = tf.Variable(tf.zeros([vocab_size]))
		logits = tf.matmul(output, w) + b

	# Softmax prediction
	with tf.variable_scope('predicion'):
		prediction = tf.nn.softmax(logits)

	# Loss/Cost
	with tf.variable_scope('cost'):
		y = tf.placeholder(tf.int32, [None, None, vocab_size])
		labels = tf.reshape(y, [-1, vocab_size])
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

	# Optimizer
	with tf.variable_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	# Tensorboard logging
	with tf.variable_scope('logging'):
	    tf.summary.scalar('current_cost', loss)
	    tf.summary.scalar('current preplexity', tf.exp(loss))
	    summary = tf.summary.merge_all()

with tf.Session(graph=graph) as sess:
	saver = tf.train.Saver(max_to_keep=2)
	try: 
		saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
		print("Restored model")
	except:
		tf.global_variables_initializer().run()
	for epoch in range(1, epochs+1):
		print("="*40 + " Epoch: " + str(epoch) + " " + "="*50)
		l = 0
		perplexity = 0
		for step in range(steps):
			seqlength, train_b = batch.nextBatch()
			feed = {x: train_b[:,:-1,:], y: train_b[:,1:,:],
			        seqlen: [seqlength for _ in range(batch_size)]}
			labels_, train_prediction_, _, loss_ = sess.run(
			    [labels, prediction, optimizer, loss], feed_dict=feed)
			l += loss_
			perplexity += float(np.exp(logprob(train_prediction_, labels_)))
			bar.update(step, loss=loss_)
		time.sleep(1)
		print("\nAverage loss: ", l/(steps+1))
		print("Average perplexity: ", perplexity/(steps+1))

		max_len = 100
		new_joke = np.zeros((1, max_len, vocab_size))
		new_joke[0,0,data.w2i(data.sentence_start_token)]=1
		sentance_length = 1
		while not np.argmax(new_joke[0,sentance_length-1,:]) == data.w2i(data.sentence_end_token) and sentance_length<max_len:
			new_wordprobs = prediction.eval({x: new_joke[:,0:sentance_length,:],
				seqlen: [sentance_length]})
			try:
				samples = np.random.multinomial(1, new_wordprobs[-1])
				sampled_word = np.argmax(samples)
			except:
				print('Sum(pvals) > 1, argmax instead.')
				sampled_word = np.argmax(new_wordprobs[-1])
			if data.i2w(sampled_word) != data.unknown_token:
				new_joke[0,sentance_length,sampled_word] = 1
				sentance_length +=1
		print()
		string = np.array([])
		for j in new_joke[0]:
			if np.count_nonzero(j) >= 1:
				string = np.append(string, data.i2w(np.argmax(j)))
		print(" ".join(string))
		print()
	saver.save(sess, './checkpoints/mymodel', global_step = epoch)


