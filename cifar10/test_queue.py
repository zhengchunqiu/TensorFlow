"""TensorFlow通过队列读取数据"""

import os
import scipy.misc
import tensorflow as tf

#从queue中返回一个样本
def read_queue(data_dir='cifar10_data/cifar-10-batches-bin'):
    #5个bin文件
    filenames=[os.path.join(data_dir,'data_batch_%d.bin')%i for i in range(1,6)]
    print(filenames)

    #使用tf.train.string_input_producer函数把我们需要的全部文件打包为一个tf内部的queue类型，
    # 之后tf开文件就从这个queue中取目录了，要注意一点的是这个函数的shuffle参数默认是True，
    # 也就是你传给他文件顺序是1234，但是到时候读就不一定了，输出无序。
    # 当shuffle=False时，输出有序。
    filename_queue=tf.train.string_input_producer(filenames,shuffle=False)

    #搞一个reader，不同reader对应不同的文件结构，比如度bin文件tf.FixedLengthRecordReader就比较好，
    # 因为每次读等长的一段数据。
    reader=tf.FixedLengthRecordReader(record_bytes=1+3*32*32)

    #用reader的read方法，这个方法需要一个IO类型的参数，就是我们上边string_input_producer输出的那个queue了，
    # reader从这个queue中取一个文件目录，然后打开它经行一次读取，reader的返回是一个tensor（这一点很重要，
    # 我们现在写的这些读取代码并不是真的在读数据，还是在画graph，和定义神经网络是一样的，
    # 这时候的操作在run之前都不会执行，这个返回的tensor也没有值，他仅仅代表graph中的一个结点）
    key,value=reader.read(filename_queue)

    #将字符串解析成图像对应的unit8像素组
    record_bytes=tf.decode_raw(value,tf.uint8)

    #tf.cast类型转换，tf.strided_slice切片[start,end)
    label = tf.cast(tf.strided_slice(record_bytes, [0], [1]), tf.int32)
    img=tf.reshape(tf.strided_slice(record_bytes,[1],[1+3*32*32]),[3,32,32])

    # Convert from [depth, height, width] to [height, width, depth].
    uint8image = tf.transpose(img, [1, 2, 0])

    image=tf.cast(uint8image,tf.float32)
    image.set_shape([32,32,3])
    label.set_shape([1])

    return image,label


#tf.train.batch是按顺序读取数据，队列中的数据始终是一个有序的队列，当
# batch_size是返回的一个batch的大小
# num_threads，当nums_threads=1是，出队是顺序的;当num_thread>1是，数据是多进程读取的，出队的数据是无序的
# capacity是队列的长度，比如capacity=10，开始队列内容为0，..,9=>读取5条记录后，队列剩下5,..,9，
# 然后又补充5条变成=>5,..,14,队头一直按顺序补充，队尾一直按顺序出队
def get_batch(image,label):
    image_batch, label_batch = tf.train.batch([image, label],
                                         batch_size=16,
                                         num_threads=128,
                                         capacity=20000+3*128)
    label_batch=tf.reshape(label_batch, [10])
    return image_batch,label_batch

image,label=read_queue()
image_batch,label_batch=get_batch(image,label)
if __name__=='__main__':
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)
        for step in range(2):
            images,labels=sess.run([image_batch,label_batch])
            print(labels)

            #print(images[0],labels[0])

            #scipy.misc.toimage(images[0]).save('0.jpg')









