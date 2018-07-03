from eurosatTS.Dataset import Dataset
import tensorflow as tf
import os


if __name__ == '__main__':

    dataset = Dataset("eurosatDb.tfrecord")

    dataset.createDataset()


