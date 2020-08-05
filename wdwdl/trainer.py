from __future__ import print_function, division
import wdwdl.hyperparameter_optimization as hpopt
import wdwdl.utils as utils
import optuna
import tensorflow as tf


def train_nn_wa_classification(args, data_set, label, preprocessor):
    """ We use an deep artificial neural network for learning the mapping of process instances and labels. """

    if args.hpopt:
        hpopt.create_data(data_set, label, preprocessor, args)

        sampler = optuna.samplers.TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(find_best_model, n_trials=args.hpopt_eval_runs)
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        return study.best_trial.number
    else:
        train_model(args, data_set, label, preprocessor)
        return -1


def find_best_model(trial):
    args = hpopt.args
    x_train = hpopt.x_train
    y_train = hpopt.y_train
    x_test = hpopt.x_test
    y_test = hpopt.y_test

    """
    # 0.) test: perceptron
    input_layer = tf.keras.layers.Input(shape=(hpopt.time_steps, hpopt.number_attributes), name='input_layer')
    input_layer_flattened = tf.keras.layers.Flatten()(input_layer)

    layer_1 = tf.keras.layers.Dense(100, activation=trial.suggest_categorical("activation", args.hpopt_activation))(
        input_layer_flattened)
    layer_2 = tf.keras.layers.Dropout(0.2)(layer_1)
    b1 = tf.keras.layers.Dropout(0.2)(layer_2)

    
    # 1.) test: lstm according to Weinzierl et al. (2020)
    input_layer = tf.keras.layers.Input(shape=(time_steps, number_attributes), name='input_layer')
    hidden_layer_1 = tf.keras.layers.recurrent.LSTM(100,
                                                 implementation=2,
                                                 kernel_initializer='glorot_uniform',  # activation="tanh"
                                                 return_sequences=False)(input_layer)
    hidden_layer_1 = tf.keras.layers.Dropout(0.2)(hidden_layer_1)
    b1 = tf.keras.layers.normalization.BatchNormalization()(hidden_layer_1)


    # 2.) test: mlp according to Theis et al. (2019)
    input_layer = tf.keras.layers.Input(shape=(time_steps, number_attributes), name='input_layer')
    input_layer_flattened = tf.keras.layers.Flatten()(input_layer)

    # layer 2
    layer_1 = tf.keras.layers.Dense(300, activation='relu')(input_layer_flattened)
    layer_1 = tf.keras.layers.normalization.BatchNormalization()(layer_1)
    layer_1 = tf.keras.layers.Dropout(0.5)(layer_1)

    # layer 3
    layer_2 = tf.keras.layers.Dense(200, activation='relu')(layer_1)
    layer_2 = tf.keras.layers.normalization.BatchNormalization()(layer_2)
    layer_2 = tf.keras.layers.Dropout(0.5)(layer_2)

    # layer 4
    layer_3 = tf.keras.layers.Dense(100, activation='relu')(layer_2)
    layer_3 = tf.keras.layers.normalization.BatchNormalization()(layer_3)
    layer_3 = tf.keras.layers.Dropout(0.5)(layer_3)

    # layer 5
    layer_4 = tf.keras.layers.Dense(50, activation='relu')(layer_3)
    layer_4 = tf.keras.layers.normalization.BatchNormalization()(layer_4)
    b1 = tf.keras.layers.Dropout(0.5)(layer_4)


    # 3.) test: custom mlp
    input_layer = tf.keras.layers.Input(shape=(data_set.shape[1], ), name='input_layer')
    layer_1 = tf.keras.layers.Dense(200, activation='tanh')(input_layer)
    layer_2 = tf.keras.layers.Dropout(0.2)(layer_1)
    layer_2 = tf.keras.layers.Dense(150, activation='tanh')(layer_2)
    layer_3 = tf.keras.layers.Dropout(0.2)(layer_2)
    layer_3 = tf.keras.layers.Dense(100, activation='tanh')(layer_3)
    layer_4 = tf.keras.layers.Dropout(0.2)(layer_3)
    layer_4 = tf.keras.layers.Dense(50, activation='tanh')(layer_4)
    b1 = tf.keras.layers.Dropout(0.2)(layer_4)


    # 4.) test: custom cnn
    input_layer = tf.keras.layers.Input(shape=(time_steps, number_attributes), name='input_layer')
    layer_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='valid', activation='relu')(input_layer)
    layer_1 = tf.keras.layers.MaxPool1D()(layer_1)
    layer_1 = tf.keras.layers.Flatten()(layer_1)
    b1 = tf.keras.layers.Dense(100, activation='relu')(layer_1)
    # b1 = tf.keras.layers.Dropout(0.2)(layer_1)
    """

    # cnn according to Abdulrhman et al. (2019)
    input_layer = tf.keras.layers.Input(shape=(hpopt.time_steps, hpopt.number_attributes), name='input_layer')
    layer_1 = tf.keras.layers.Conv1D(filters=128,
                                     kernel_size=16,
                                     padding='same',
                                     strides=1,
                                     activation=trial.suggest_categorical('activation', args.hpopt_activation))(input_layer)
    layer_2 = tf.keras.layers.MaxPool1D()(layer_1)
    layer_2 = tf.keras.layers.Conv1D(filters=128,
                                     kernel_size=16,
                                     padding='same',
                                     strides=1,
                                     activation=trial.suggest_categorical('activation', args.hpopt_activation))(layer_2)
    layer_3 = tf.keras.layers.MaxPool1D()(layer_2)
    layer_3 = tf.keras.layers.Conv1D(filters=128,
                                     kernel_size=16,
                                     padding='same',
                                     strides=1,
                                     activation=trial.suggest_categorical('activation', args.hpopt_activation))(layer_3)
    layer_4 = tf.keras.layers.MaxPool1D()(layer_3)
    layer_4 = tf.keras.layers.Conv1D(filters=128,
                                     kernel_size=16,
                                     padding='same',
                                     strides=1,
                                     activation=trial.suggest_categorical('activation', args.hpopt_activation))(layer_4)
    layer_4 = tf.keras.layers.MaxPool1D()(layer_4)
    layer_5 = tf.keras.layers.Conv1D(filters=128,
                                     kernel_size=16,
                                     padding='same',
                                     strides=1,
                                     activation='relu')(layer_4)
    layer_5 = tf.keras.layers.MaxPool1D()(layer_5)
    layer_6 = tf.keras.layers.Flatten()(layer_5)
    b1 = tf.keras.layers.Dense(100, activation='relu')(layer_6)


    output = tf.keras.layers.Dense(y_train.shape[1],
                                   activation='softmax',
                                   name='output',
                                   kernel_initializer='glorot_uniform')(b1)
    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output])


    # optimizer = tf.keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004, clipvalue=3)
    # optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam',
                  metrics=['accuracy', utils.f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        '%sclf_wa_mapping_trial%s.h5' % (args.checkpoint_dir, trial.number),
        monitor='val_loss',
        verbose=0,
        save_weights_only=False,   # save_best_only=True,
        mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                      mode='auto',
                                                      min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()
    model.fit(x_train, {'output': y_train}, validation_split=0.1, verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1]


def train_model(args, data_set, label, preprocessor):
    """
    Trains a model for activity prediction without HPO (for test purposes during development).
    HPOs are the same as reported in the ECIS2020 Paper.
    """

    number_attributes = preprocessor.data_structure['encoding']['num_values_features'] - 2  # case + time
    time_steps = preprocessor.data_structure['meta']['max_length_process_instance']
    data_set = data_set.reshape((data_set.shape[0], time_steps, number_attributes))

    # 0.) test: perceptron
    input_layer = tf.keras.layers.Input(shape=(time_steps, number_attributes), name='input_layer')
    input_layer_flattened = tf.keras.layers.Flatten()(input_layer)

    layer_1 = tf.keras.layers.Dense(100, activation='tanh')(input_layer_flattened)
    layer_2 = tf.keras.layers.Dropout(0.2)(layer_1)
    b1 = tf.keras.layers.Dropout(0.2)(layer_2)

    """
    # 1.) test: lstm according to Weinzierl et al. (2020)
    input_layer = tf.keras.layers.Input(shape=(time_steps, number_attributes), name='input_layer')
    hidden_layer_1 = tf.keras.layers.recurrent.LSTM(100,
                                                 implementation=2,
                                                 kernel_initializer='glorot_uniform',  # activation="tanh"
                                                 return_sequences=False)(input_layer)
    hidden_layer_1 = tf.keras.layers.Dropout(0.2)(hidden_layer_1)
    b1 = tf.keras.layers.normalization.BatchNormalization()(hidden_layer_1)
    
    
    # 2.) test: mlp according to Theis et al. (2019)
    input_layer = tf.keras.layers.Input(shape=(time_steps, number_attributes), name='input_layer')
    input_layer_flattened = tf.keras.layers.Flatten()(input_layer)

    # layer 2
    layer_1 = tf.keras.layers.Dense(300, activation='relu')(input_layer_flattened)
    layer_1 = tf.keras.layers.normalization.BatchNormalization()(layer_1)
    layer_1 = tf.keras.layers.Dropout(0.5)(layer_1)

    # layer 3
    layer_2 = tf.keras.layers.Dense(200, activation='relu')(layer_1)
    layer_2 = tf.keras.layers.normalization.BatchNormalization()(layer_2)
    layer_2 = tf.keras.layers.Dropout(0.5)(layer_2)

    # layer 4
    layer_3 = tf.keras.layers.Dense(100, activation='relu')(layer_2)
    layer_3 = tf.keras.layers.normalization.BatchNormalization()(layer_3)
    layer_3 = tf.keras.layers.Dropout(0.5)(layer_3)

    # layer 5
    layer_4 = tf.keras.layers.Dense(50, activation='relu')(layer_3)
    layer_4 = tf.keras.layers.normalization.BatchNormalization()(layer_4)
    b1 = tf.keras.layers.Dropout(0.5)(layer_4)

    
    # 3.) test: custom mlp
    input_layer = tf.keras.layers.Input(shape=(data_set.shape[1], ), name='input_layer')
    layer_1 = tf.keras.layers.Dense(200, activation='tanh')(input_layer)
    layer_2 = tf.keras.layers.Dropout(0.2)(layer_1)
    layer_2 = tf.keras.layers.Dense(150, activation='tanh')(layer_2)
    layer_3 = tf.keras.layers.Dropout(0.2)(layer_2)
    layer_3 = tf.keras.layers.Dense(100, activation='tanh')(layer_3)
    layer_4 = tf.keras.layers.Dropout(0.2)(layer_3)
    layer_4 = tf.keras.layers.Dense(50, activation='tanh')(layer_4)
    b1 = tf.keras.layers.Dropout(0.2)(layer_4)

    
    # 4.) test: custom cnn
    input_layer = tf.keras.layers.Input(shape=(time_steps, number_attributes), name='input_layer')
    layer_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='valid', activation='relu')(input_layer)
    layer_1 = tf.keras.layers.MaxPool1D()(layer_1)
    layer_1 = tf.keras.layers.Flatten()(layer_1)
    b1 = tf.keras.layers.Dense(100, activation='relu')(layer_1)
    # b1 = tf.keras.layers.Dropout(0.2)(layer_1)
    
    
    # cnn according to Abdulrhman et al. (2019)
    input_layer = tf.keras.layers.Input(shape=(time_steps, number_attributes), name='input_layer')
    layer_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(input_layer)
    layer_2 = tf.keras.layers.MaxPool1D()(layer_1)
    layer_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(layer_2)
    layer_3 = tf.keras.layers.MaxPool1D()(layer_2)
    layer_3 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(layer_3)
    layer_4 = tf.keras.layers.MaxPool1D()(layer_3)
    layer_4 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(layer_4)
    layer_4 = tf.keras.layers.MaxPool1D()(layer_4)
    layer_5 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(layer_4)
    layer_5 = tf.keras.layers.MaxPool1D()(layer_5)
    layer_6 = tf.keras.layers.Flatten()(layer_5)
    b1 = tf.keras.layers.Dense(100, activation='relu')(layer_6)
    """

    output = tf.keras.layers.core.Dense(label.shape[1], activation='softmax', name='output',
                                        kernel_initializer='glorot_uniform')(b1)
    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output])

    # optimizer = tf.keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004, clipvalue=3)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(loss={'output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%sclf_wa_mapping.h5' % args.checkpoint_dir,
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                      mode='auto',
                                                      min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()
    model.fit(data_set, {'output': label}, validation_split=0.1, verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)
