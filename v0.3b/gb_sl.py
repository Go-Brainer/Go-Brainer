# tag::alphago_sl_data[]
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.agent.predict import DeepLearningAgent
from dlgo.networks.gb_net import gb_model

from keras.callbacks import ModelCheckpoint
from multiprocessing import freeze_support
import h5py
import tensorflow as tf
from keras.optimizers import Adadelta, Adam, SGD

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

    rows, cols = 19, 19
    num_classes = rows * cols
    num_games = 4000

    encoder = SevenPlaneEncoder((rows,cols))
    processor = GoDataProcessor(encoder=encoder.name())
    generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_games, use_generator=True)
    # end::alphago_sl_data[]

    # tag::alphago_sl_model[]
    input_shape = (encoder.num_planes, rows, cols)
    gb_sl = gb_model(input_shape, is_policy_net=True)

    opt = SGD(lr=0.005, momentum=0.95)
    gb_sl.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # end::alphago_sl_model[]

    # tag::alphago_sl_train[]
    epochs = 20
    batch_size = 512
    gb_sl.fit_generator(
        generator=generator.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=generator.get_num_samples() / batch_size,
        validation_data=test_generator.generate(batch_size, num_classes),
        validation_steps=test_generator.get_num_samples() / batch_size,
        callbacks=[ModelCheckpoint('./agents/chk/gb_sl_epch{epoch}.h5'), ]
    )

    alphago_sl_agent = DeepLearningAgent(gb_sl, encoder)

    with h5py.File('gb_sl.h5', 'w') as sl_agent_out:
        alphago_sl_agent.serialize(sl_agent_out)
    # end::alphago_sl_train[]

    gb_sl.evaluate_generator(
        generator=test_generator.generate(batch_size, num_classes),
        steps=test_generator.get_num_samples() / batch_size
    )

if __name__ == '__main__':
    freeze_support()
    main()