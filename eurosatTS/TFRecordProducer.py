import rasterio
import os
import numpy as np
import tensorflow as tf


def parseFilenames(path):
    # -1 cause it starts from the root dir ('/train' or '/test' ;) )

    data = []

    i = -1
    for dirName, subdirList, fileList in os.walk(path):
        # print('\t%s' % dirName)
        # if dirName.endswith("AnnualCrop"):
        for fname in fileList:
            if fname.endswith("AnnualCrop_1.tif") or fname.endswith("AnnualCrop_2.tif") or fname.endswith("AnnualCrop_3.tif"):
                with rasterio.open(os.path.join(dirName, fname)) as src:
                    image = np.transpose(src.read([4,3,2]),[1,2,0])
                    # g = src.read(3)
                    # b = src.read(2)
                    # # Numpy Img convention: HxWxC
                    # rgb = np.array(image, dtype=np.float32).reshape((64, 64, 3))
                    data.append((image, i))
        i += 1
    return data

if __name__ == '__main__':
    data = parseFilenames("Dataset/eurosat_prova/train")
    writer = tf.python_io.TFRecordWriter('eurosatDb.tfrecord')

    for item in data:
        print(item[0].shape)
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
