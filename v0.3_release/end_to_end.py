# tag::e2e_imports[]
import h5py
import time

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder as XPlaneEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import large
from multiprocessing import freeze_support
# end::e2e_imports[]

def main():

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_soft_device_placement(True)
        except RuntimeError as e:
            print(e)

    samp = 2500
    epo = 20


    # tag::e2e_processor[]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.isdir("./agents"):
        os.makedirs("./agents")
    model_h5filename = './agents/deep_bot_B_%d.h5'

    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    encoder = XPlaneEncoder((go_board_rows, go_board_cols))
    data_dir = "data/" + encoder.name()
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    processor = GoDataProcessor(encoder=encoder.name(), data_directory=data_dir)


    # end::e2e_processor[]

    # tag::e2e_model[]
    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    network_layers = large.layers(input_shape)
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


    X, y = processor.load_go_data(num_samples=samp)
    model.fit(X, y, batch_size=128, epochs=epo, verbose=1)
    # end::e2e_model[]

    # tag::e2e_agent[]
    deep_learning_bot = DeepLearningAgent(model, encoder)
    deep_learning_bot.serialize(h5py.File(model_h5filename % (0,), "w"))
    # end::e2e_agent[]

            # # tag::e2e_load_agent[]
            # model_file = h5py.File(model_h5filename, "r")
            # bot_from_file = load_prediction_agent(model_file)
            #
            # web_app = get_web_app({'predict': bot_from_file})
            # web_app.run()
            # end::e2e_load_agent[]

    # model_file = h5py.File("./agents/deep_bot_A0.h5", "r")
    # bot_from_file = load_prediction_agent(model_file)
    #
    # web_app = get_web_app({'predict': bot_from_file})
    # web_app.run()

if __name__ == '__main__':
    freeze_support()
    main()

