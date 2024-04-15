import numpy as np
import os
import time
from datetime import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import ReLU
from keras.utils import plot_model
import cv2
from utils import define_model, prepare_dataset, unetpp
from keras import regularizers
import numpy as np
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
def train(iteration=0, DATASET='DRIVE', crop_size=128, need_au=True, ACTIVATION='ReLU', dropout=0.1, batch_size=32,
          repeat=4, minimum_kernel=32, epochs=200):
    # repeat 表示数据增强的重复次数
    model_name = f"Final_crop_size_{crop_size}_epochs_{epochs}"

    print("Model : %s" % model_name)

    prepare_dataset.prepareDataset(DATASET)

    activation = globals()[ACTIVATION]
    model = define_model.get_Attention()





    try:
        os.makedirs(f"trained_model/{DATASET}/", exist_ok=True)
        os.makedirs(f"logs/{DATASET}/", exist_ok=True)
    except:
        pass

    # load_path = f"trained_model/{DATASET}/{model_name}_weights.best.hdf5"
    # load_path = f"/root/autodl-fs/IterNet/IterNet/trained_model/DRIVE/(res3+spa+1.5g)Final_crop_size_128_epochs_100.hdf5"
    try:
        model.load_weights(load_path, by_name=True)
    except:
        pass

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d---%H-%M-%S")

    tensorboard = TensorBoard(
        log_dir=f"logs/{DATASET}/Final_Emer-Cropsize_{crop_size}-Epochs_{epochs}---{date_time}",
        histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
        write_images=True, embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    save_path = f"trained_model/{DATASET}/{model_name}.hdf5"
    # checkpoint = ModelCheckpoint(save_path, monitor='final_out_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint = ModelCheckpoint(save_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # data_generator = define_model.Generator(batch_size, repeat, DATASET)
    data_generator = unetpp.Generator(batch_size, repeat, DATASET)

    history = model.fit(data_generator.gen(au=need_au, crop_size=crop_size, iteration=iteration),
                                  epochs=epochs, verbose=1,
                                  steps_per_epoch=100 * data_generator.n // batch_size,
                                  use_multiprocessing=True, workers=8,
                                  callbacks=[tensorboard, checkpoint])
if __name__ == "__main__":
    train(DATASET='CHASEDB1',  # DRIVE, CHASEDB1 or STARE
          batch_size=16, epochs=100)

