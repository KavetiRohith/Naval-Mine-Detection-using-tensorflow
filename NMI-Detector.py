import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#reading the dataset
def read_dataset():
    df=pd.read_csv("E:\\PycharmProjects\\tf edureka\\sonar.csv")
    ##here the DataFrame.values converts the pandas dataframe into
    ##a numpy array
    X=df[df.columns[0:60]].values
    y=df[df.columns[60]]
    ##encoding the labels
    encoder=LabelEncoder()
    encoder.fit(y)
    y=encoder.transform(y)
    Y=one_hot_encode(y)
    print(X.shape)
    return (X,Y)

##define the one hot encoder function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode =np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels]=1
    return one_hot_encode

# read the dataset using read_dataset function
X,Y = read_dataset()

#Shuffle the dataset to mix up the rows
X,Y = shuffle(X,Y, random_state=1)

#split the dataset into training and testing adatset using
# train_tset_split method in sklearn.model_selection
train_x , test_x , train_y , test_y = train_test_split(X, Y, test_size=0.20 ,random_state= 20)

#define the important parameters and variables to work with tensors
learning_rate = 0.3
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)
n_dim=X.shape[1]
print("n_dim",n_dim)
n_class=2
model_path="E:\\PycharmProjects\\tf edureka"


n_hidden_1=60
n_hidden_2=60
n_hidden_3=60
n_hidden_4=60

x=tf.placeholder(tf.float32,[None,n_dim])
W=tf.Variable(tf.zeros([n_dim, n_class]))
b=tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32,[None, n_class])

#define the model
def multilayer_perceptron(x, weights, biases):
    # hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
    layer_3 = tf.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3,weights['h4']),biases['b4'])
    layer_4 = tf.sigmoid(layer_4)

    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

weights = {
'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
'h2': tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}

biases={
'b1':tf.Variable(tf.truncated_normal([n_hidden_1])),
'b2':tf.Variable(tf.truncated_normal([n_hidden_2])),
'b3':tf.Variable(tf.truncated_normal([n_hidden_3])),
'b4':tf.Variable(tf.truncated_normal([n_hidden_4])),
'out':tf.Variable(tf.truncated_normal([n_class]))
}

# initialize all the variables

init = tf.global_variables_initializer()

saver = tf.train.Saver()

#define your model
y = multilayer_perceptron(x, weights, biases)

#define the cost function and optimizer
cost_function =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step =  tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

#calculate the cost and accuracy for each epoch

mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x: train_x,y_: train_y})
    cost = sess.run(cost_function,feed_dict={x: train_x,y_: train_y})
    cost_history= np.append(cost_history,cost)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # print("accuracy: " ,sess.run(accuracy,feed_dict={x: train_x,y_: train_y}))
    pred_y = sess.run(y,feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = (sess.run(accuracy,feed_dict={x: train_x,y_: train_y}))
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy,feed_dict={x: train_x,y_: train_y}))
    accuracy_history.append(accuracy)

    print('epoch : ',epoch , ' - ','cost: ',cost ,"-MSE: ",mse_,"-train_accuracy",accuracy)

save_path = saver.save(sess, model_path)
print("MOdel saved in file: %s" % save_path)

# plot mse and accuracy graph

plt.plot(mse_history, 'r')
plt.show()
plt.plot(accuracy_history)
plt.show()

# print the final accuracy

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ",(sess.run(accuracy, feed_dict={x: test_x,y_:test_y})))

#print final mean squared error

pred_y = sess.run(y,feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y-test_y))
print("MSE: %.4f"% sess.run(mse))