import numpy as np
import tensorflow as tf


x = tf.range(1, 10, name="x")
print (tf.contrib.learn.run_n({"x": x}, n=1, feed_dict=None))
# y = tf.Variable([[1,2,3],[2,3,4,5], [1]])
# print (tf.contrib.learn.run_n({"y": y}, n=1, feed_dict=None))

# # A queue that outputs 0,1,2,3,...
range_q = tf.train.range_input_producer(limit=5, shuffle=False)

# slice_end = range_q.dequeue()
# print (tf.contrib.learn.run_n({"slice_end": slice_end}, n=1, feed_dict=None))
 
# # Slice x to variable length, i.e. [0], [0, 1], [0, 1, 2], ....
# y = tf.slice(x, [0], [slice_end], name="y")
# print (tf.contrib.learn.run_n({"y": y}, n=1, feed_dict=None))
 
# # Batch the variable length tensor with dynamic padding
y = tf.Variable([[1,2,3,4],[1,2,3],[1,7,8,5]])
# y = tf.Variable([[tf.constant(n) for n in range(1,5)], [tf.constant(n) for n in range(1,2)]])
# with tf.Session() as sess:
#     tf.initialize_all_variables()
#     pp = sess.run(y)
#     print (p)


# print (tf.contrib.learn.run_n({"y": y}, n=1, feed_dict=None))
batched_data = tf.train.batch(
    tensors=[y],
    batch_size=3,
    dynamic_pad=True,
    name="y_batch"
)


# var = tf.Variable([1,2,3,4,5]) 
# # Run the graph
# # tf.contrib.learn takes care of starting the queues for us
# res = tf.contrib.learn.run_n({"y": batched_data}, n=1, feed_dict=None)
 
# # Print the result
# print("Batch shape: {}".format(res[0]["y"].shape))
# print (tf.contrib.learn.run_n({"y": batched_data}, n=1, feed_dict=None))
# print(res[0]["y"])

# # tf.initialize_all_variables()
# # with tf.Session() as sess:
# #     # sses.initialize_all_variables()
# #     a = sess.run(x)
# #     b = sess.run(y)
# #     print (b)

# # tf.contrib.learn.run_n(output_dict, feed_dict=None, restore_checkpoint_path=None, n=1)






################################################################
################################################################
################################################################




import numpy as np
import tensorflow as tf

sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]

corpora_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/corpus.tfrecords'

# 
def make_TF_sequences(input_sequence, label_sequence):
    row_seq = tf.train.SequenceExample()
    seqlen = len(input_sequence)
    print (seqlen)
    row_seq.context.feature["length"].int64_list.value.append(seqlen)
    # Defining Feature list for for the tokens and the labels. 
    tokens_fl =  row_seq.feature_lists.feature_list["tokens"]
    labels_fl =  row_seq.feature_lists.feature_list["labels"]

    for token, label in zip(input_sequence, label_sequence):
        # Adding all the tokens and labels for one training row into feature lists
        tokens_fl.feature.add().int64_list.value.append(token)  
        labels_fl.feature.add().int64_list.value.append(label)
    return  row_seq


########### Write Data into tfrecords #############
# Write all the examples into a TFRecords file and store it into the disk. One could also simply store the numpy arrays into the disk but it comes with many benefits and we would like to utilize them.
def write_records():
    rec_writer = tf.python_io.TFRecordWriter(corpora_dir)
    for sequence, label_sequence in zip(sequences, label_sequences):
        print (sequence)
        rec_seq = make_TF_sequences(sequence, label_sequence)
        # Write each record sequence into the disk using TFRecordWriter.
        rec_writer.write(rec_seq.SerializeToString()) 


########### Read sequence Data from Disk #############
# a = tf.TFRecordReader(corpora_dir)

# a = tf.Print(serialized_example, [serialized_example], message="This is the serialized_example: ")

def read_decode_single_example(file_name):
    filename_queue = tf.train.string_input_producer([file_name], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'tokens': tf.VarLenFeature([], tf.int64),
            'image': tf.VarLenFeature([], tf.int64)
        })
    # now return the converted data
    label = features['label']
    image = features['image']
    return label, image




# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init_op) #execute init_op
#     # sess.run(a)
#     print (sess.run(serialized_example))




# sess.close()
# a.eval()
# with tf.Session() as sess:
#     sess.run(init_op) #execute init_op
#     #print the random values that we sample
#     print (sess.run(serialized_example))






## FOR RNN we just need one graph since all the sequence graph are just a replica of each other where they receive input from the pevious state:
# We can then execute our graph for each time step, feeding in the state returned from the previous execution into the current execution. This would work for a model that was already trained, but there’s a problem with using this approach for training: the gradients computed during backpropagation are graph-bound. We would only be able to backpropagate errors to the current timestep; we could not backpropagate the error to time step t-1. This means our network will not be able to learn how to store long-term dependencies (such as the two in our data) in its state.
# Alternatively, we might make our graph as wide as our data sequence. This often works, except that in this case, we have an arbitrarily long input sequence, so we have to stop somewhere. Let’s say we make our graph accept sequences of length 10,000. This solves the problem of graph-bound gradients, and the errors from time step 9999 are propagated all the way back to time step 0. Unfortunately, such backpropagation is not only (often prohibitively) expensive, but also ineffective, due to the vanishing / exploding gradient problem: it turns out that backpropagating errors over too many time steps often causes them to vanish (become insignificantly small) or explode (become overwhelmingly large).




