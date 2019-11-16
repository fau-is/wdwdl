from __future__ import print_function, division
import keras


def train_nn_wa_classification(args, data_set, label, preprocessor):
    """ We use an deep artificial neural network for learning the mapping of process instances and the label. """


    number_attributes = preprocessor.data_structure['encoding']['num_values_features'] - 2  # case + time
    time_steps = preprocessor.data_structure['meta']['max_length_process_instance']

    data_set = data_set.reshape((data_set.shape[0], time_steps, number_attributes))

    input_layer = keras.layers.Input(shape=(time_steps, number_attributes), name='input_layer')
    layer_1 = keras.layers.Conv1D(filters=128, kernel_size=2, padding='valid', activation='relu')(input_layer)
    # layer_1 = keras.layers.GlobalAveragePooling1D(pool_size=2)(layer_1)
    layer_1 = keras.layers.Flatten()(layer_1)
    b1 = keras.layers.Dense(100, activation='relu')(layer_1)
    #b1 = keras.layers.Dropout(0.2)(layer_1)



    """
    input_layer = keras.layers.Input(shape=(data_set.shape[1], ), name='input_layer')
    layer_1 = keras.layers.Dense(200, activation='tanh')(input_layer)
    layer_2 = keras.layers.Dropout(0.2)(layer_1)
    layer_2 = keras.layers.Dense(150, activation='tanh')(layer_2)
    layer_3 = keras.layers.Dropout(0.2)(layer_2)
    layer_3 = keras.layers.Dense(100, activation='tanh')(layer_3)
    layer_4 = keras.layers.Dropout(0.2)(layer_3)
    layer_4 = keras.layers.Dense(50, activation='tanh')(layer_4)
    b1 = keras.layers.Dropout(0.2)(layer_4)
    """

    output = keras.layers.core.Dense(label.shape[1], activation='softmax', name='output',
                                           kernel_initializer='glorot_uniform')(b1)

    model = keras.models.Model(inputs=[input_layer], outputs=[output])

    optimizer = keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004, clipvalue=3)
    # optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss={'output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    model_checkpoint = keras.callbacks.ModelCheckpoint('%sclf_wa_mapping.h5' % args.checkpoint_dir,
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto')
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                                   min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()
    model.fit(data_set, {'output': label}, validation_split=0.1, verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

