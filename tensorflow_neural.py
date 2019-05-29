import tensorflow as tf

from numpy.random import RandomState

batch_Size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 5], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None,2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,5), name='y-input')

#神经网络的前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
    y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    +(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10, 1.0))
)
#定义损失函数
#loss = tf.nn.softmax_cross_entropy_with_logits(label= ,logits=)#封装交叉熵和softmax

#损失函数的优化算法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#学习率的设计--指数衰减法
#decay_learning_rate = learning_rate*decay_rate^(global_steps/dacay_steps)阶梯函数

#exponential_decay生成学习率
global_steps = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1,global_steps,100,0.96,staircase=True)

#使用指数衰减率的学习率，在minimize函数中传入global_step,从而更新学习率
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=,global_step=global_steps)


rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
print(X)

Y = [[int(x1+x2 <1)] for (x1, x2)in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 100
    for i in range(STEPS):
        start =(i*batch_Size)%dataset_size
        end = min(start+batch_Size,dataset_size)

        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start,end]})
        if i%10 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print('after %d training steps, cross_entropy on all data is %g' % i, total_cross_entropy)




