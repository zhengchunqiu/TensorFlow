import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 该函数将给出权重初始化的方法
def weight_init(shape,stddev=0.1):
    initial=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    return initial

# 定义输入矩阵的占位符，输入层单元为784，None代表批量大小的占位，X代表输入的真实图片。占位符的数值类型为32位浮点型
X=tf.placeholder(tf.float32,[None,784])

# 定义判别器的权重矩阵和偏置项向量，由此可知判别网络为三层全连接网络
d_w1=weight_init(shape=[784,128])
d_b1=tf.Variable(tf.constant(0.0,shape=[128]))
d_w2=weight_init(shape=[128,1])
d_b2=tf.Variable(tf.constant(0.0,shape=[1]))
theta_d=[d_w1,d_w2,d_b1,d_b2]

# 定义生成器的输入噪声为100维度的向量组，None根据批量大小确定
z=tf.placeholder(tf.float32,[None,100])

# 定义生成器的权重与偏置项。输入层为100个神经元且接受随机噪声，
# 输出层为784个神经元，并输出手写字体图片。生成网络根据原论文为三层全连接网络
g_w1=weight_init(shape=[100,128])
g_b1=tf.Variable(tf.constant(0.0,shape=[128]))
g_w2=weight_init(shape=[128,784])
g_b2=tf.Variable(tf.constant(0.0,shape=[784]))
theta_g=[g_w1,g_w2,g_b1,g_b2]

# 定义一个可以生成m*n阶随机矩阵的函数，该矩阵的元素服从均匀分布，随机生成的z就为生成器的输入
def sample_z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])

# 定义生成器
def generator(z):
    # 第一层先计算 y=z*g_w1+g_b1,然后投入激活函数计算g_h1=ReLU（y）,g_h1 为第二次层神经网络的输出激活值
    g_h1=tf.nn.relu(tf.matmul(z,g_w1)+g_b1)
    # 以下两个语句计算第二层传播到第三层的激活结果，第三层的激活结果是含有784个元素的向量，该向量转化28×28就可以表示图像
    g_h2=tf.matmul(g_h1,g_w2)+g_b2
    g_prob=tf.nn.sigmoid(g_h2)
    return g_prob

# 定义判别器
def discriminator(x):
    d_h1=tf.nn.relu(tf.matmul(x,d_w1)+d_b1)
    d_logit=tf.matmul(d_h1,d_w2)+d_b2
    d_prob=tf.nn.sigmoid(d_logit)
    return d_prob,d_logit

# 该函数用于输出生成图片
def plot(samples):
    fig=plt.figure(figsize=(4,4))
    gs=gridspec.GridSpec(4,4)
    gs.update(wspace=0.05,hspace=0.05)

    for i,sample in enumerate(samples):
        ax=plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28),cmap='Greys_r')
    return fig

# 输入随机噪声而生成输出样本
g_sample=generator(z)

# 分别输入真实图片和生成图片，并输入判断器以判断真伪
d_real,d_logit_real=discriminator(X)
d_fake,d_logit_fake=discriminator(g_sample)

# 我们使用交叉熵作为判别器和生成器的损失函数，因为sigmoid_cross_entropy_with_logits内部会对预测输入执行Sigmoid函数，
# 所以我们取判别器最后一层未投入激活函数的值，即D_h1*D_W2+D_b2。
# tf.ones_like(D_logit_real)创建维度和D_logit_real相等的全是1的标注，真实图片。
d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real,labels=tf.ones_like(d_logit_real)))
d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake,labels=tf.zeros_like(d_logit_fake)))
d_loss=d_loss_real+d_loss_fake

# 同样使用交叉熵构建生成器损失函数
g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake,labels=tf.ones_like(d_logit_fake)))

# 定义判别器和生成器的优化方法为Adam算法，关键字var_list表明最小化损失函数所更新的权重矩阵
d_solver=tf.train.AdamOptimizer().minimize(d_loss, var_list=theta_d)
g_solver=tf.train.AdamOptimizer().minimize(g_loss, var_list=theta_g)

# 选择训练的批量大小和随机生成噪声的维度
mb_size = 128
z_dim = 100

# 读取数据集MNIST，并放在当前目录data文件夹下MNIST文件夹中，如果该地址没有数据，则下载数据至该文件夹
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

if not os.path.exists('out/'):
    os.makedirs('out/')

# 初始化，并开始迭代训练,10W次
max_steps=100000
i=0
for step in range(max_steps):
    # 每1000次输出一张生成图片
    if step%2000==0:
        samples=sess.run(g_sample,feed_dict={z:sample_z(16,z_dim)})
        fig=plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)),bbox_inches='tight')
        i+=1
        plt.close(fig)

    # next_batch抽取下一个批量的图片，该方法返回一个矩阵，即shape=[mb_size，784]，每一行是一张图片，共批量大小行
    X_mb,_=mnist.train.next_batch(mb_size)


    _,d_loss_curr=sess.run([d_solver,d_loss],feed_dict={X:X_mb,z:sample_z(mb_size,z_dim)})
    _,g_loss_curr=sess.run([g_solver,g_loss],feed_dict={z: sample_z(mb_size, z_dim)})

    # 每迭代2000次输出迭代数，生成器和判别器损失
    if step%2000==0:
        print('Iter:{}'.format(step))
        print('d_loss:{:.4}'.format(d_loss_curr))
        print('g_loss:{:.4}'.format(g_loss_curr))
        print()


