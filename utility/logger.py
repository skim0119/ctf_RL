# Module contains any methods, class, parameters, etc that is related to logging the trainig

import io

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

def record(item, writer, step):
    summary = tf.Summary()
    for key, value in item.items():
        summary.value.add(tag=key, simple_value=value)
    writer.add_summary(summary, step)
    writer.flush()

def tb_log_histogram(data, tag, step, **kargs):
    tf.summary.histogram(name=tag, data=data, step=step, **kargs)

def tb_log_ctf_frame(frame, tag, step):
    num_images = frame.shape[2]
    fig = plt.figure(1)
    ncol = 6
    nrow = (num_images//ncol)+1
    scale = 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*scale, nrow*scale))
    for n, ax in zip(range(num_images), axs.ravel()):
        image = frame[:,:,n]
        im = ax.imshow(image)
        ax.set_title('ch {0}'.format(n))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    tf.summary.image(tag, image, step=step)
