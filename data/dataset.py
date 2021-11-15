import tensorflow as tf
from absl.flags import FLAGS
from .label_anchor import AnchorLabeler
IMAGE_FEATURE_MAP = {

    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),

}

def parse_tfrecord(tfrecord, class_table, size):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    image = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    image = tf.image.resize(image, (size, size))

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    label = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

 #   paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
 #   y_train = tf.pad(y_train, paddings)
 #   y_train = anchor_label.encode(y_train)
    return image, label



def load_tfrecord_dataset(batch_size,file_pattern, class_file, img_size=640):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    
    
    dataset =  dataset.map(lambda x: parse_tfrecord(x, class_table, img_size))
 #   dataset = dataset.padded_batch(batch_size=batch_size,
  #                             padded_shapes=([None, None, 3],
 #                                             [None, 5])
  #                            )
    return dataset