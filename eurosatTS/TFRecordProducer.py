import rasterio
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def parseFilenames(path):
    # -1 cause it starts from the root dir ('/train' or '/test' ;) )

    data = []

    i = -1
    for dirName, subdirList, fileList in sorted(os.walk(path)):
        # print('\t%s' % dirName)
        # if dirName.endswith("AnnualCrop"):
        print(dirName)
        print(i)
        for fname in fileList:
            if fname.endswith(".tif"):
                with rasterio.open(os.path.join(dirName, fname)) as src:
                    image = np.transpose(src.read([4,3,2]),[1,2,0])
                    #print image.astype("float32")
                    imax = image.max()
                    imin =  image.min()
                    image = image.astype("float32")
                    image = (image -imin)/(imax-imin)
                    data.append((image, i))
        i += 1

    return data

if __name__ == '__main__':
    data = parseFilenames("dataset/eurosat_prova/train")
    writer = tf.python_io.TFRecordWriter('eurosatDb.tfrecord')

    for item in data:
        image = tf.train.Feature(bytes_list=tf.train.BytesList(value=[item[0].tostring()]))
        label = tf.train.Feature(int64_list=tf.train.Int64List(value=[np.array(item[1]).astype("int64")]))

        eurosat_dict = {
            'image': image,
            'label': label
        }

        example = tf.train.Example(features=tf.train.Features(feature=eurosat_dict))

        # Write TFrecord file# Write
        writer.write(example.SerializeToString())
    writer.close()
