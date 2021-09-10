from __future__ import print_function, division
import wdwdl.src.hyperparameter_optimization as hpo
import wdwdl.src.utils.metric as metric
import wdwdl.src.utils.general as general
import optuna
import tensorflow as tf
import sklearn


def train_ae_noise_removing(args, features_data, preprocessor):
    if args.hpo_ae:
        hpo.create_data(features_data, features_data, preprocessor, args, is_ae=True)

        if args.seed:
            sampler = optuna.samplers.TPESampler(seed=args.seed_val)  # Make the sampler behave in a deterministic way.
        else:
            sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(optimize_ae, n_trials=args.hpo_eval_runs)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        general.add_to_file(args, "hyper_params", trial, is_ae=True)
        return study.best_trial.number
    else:
        input_dimension = features_data.shape[1]
        encoding_dimension = 128

        input_layer = tf.keras.layers.Input(shape=(input_dimension,))
        encoder = tf.keras.layers.Dense(int(encoding_dimension), activation='tanh')(input_layer)
        encoder = tf.keras.layers.Dense(int(encoding_dimension / 2), activation='tanh')(encoder)
        encoder = tf.keras.layers.Dense(int(encoding_dimension / 4), activation='tanh')(encoder)
        decoder = tf.keras.layers.Dense(int(encoding_dimension / 2), activation='tanh')(encoder)
        decoder = tf.keras.layers.Dense(int(encoding_dimension), activation='tanh')(decoder)
        decoder = tf.keras.layers.Dense(input_dimension, activation='sigmoid')(decoder)

        autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
        autoencoder.summary()
        autoencoder.compile(optimizer='adam', loss='mse')

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%sae.h5' % args.checkpoint_dir,
                                                              monitor='val_loss',
                                                              verbose=0,
                                                              save_best_only=True,
                                                              save_weights_only=False,
                                                              mode='auto')
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                          mode='auto',
                                                          min_delta=0.0001, cooldown=0, min_lr=0)

        autoencoder.fit(features_data, features_data,
                        epochs=args.dnn_num_epochs_auto_encoder,
                        batch_size=args.batch_size_train,
                        shuffle=args.shuffle,  # shuffle instances per epoch
                        validation_split=args.val_model_split,
                        callbacks=[early_stopping, model_checkpoint, lr_reducer]
                        )
        return -1


def optimize_ae(trial):
    args = hpo.args
    x_train = hpo.x_train
    y_train = hpo.y_train  # identical to x_train
    x_test = hpo.x_test
    y_test = hpo.y_test  # identical to x_test

    input_dimension = x_train.shape[1]
    encoding_dimension = trial.suggest_categorical('hpo_ae_enc_dim', args.hpo_ae_enc_dim)

    if trial.suggest_categorical('hpo_ae_layers', args.hpo_ae_layers) == 1:
        # Hidden layer
        input_layer = tf.keras.layers.Input(shape=(input_dimension,))
        # Encoder
        encoder = tf.keras.layers.Dense(int(encoding_dimension),
                                        activation=trial.suggest_categorical('activation_enc_1', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical('kernel_initializer_enc_1',
                                                                                     args.hpo_kernel_initializer))(input_layer)
        # Output layer
        decoder = tf.keras.layers.Dense(input_dimension,
                                        activation=trial.suggest_categorical('activation_dec_1', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical('kernel_initializer_dec_1',
                                                                                 args.hpo_kernel_initializer)
                                                                             )(encoder)

    elif trial.suggest_categorical('hpo_ae_layers', args.hpo_ae_layers) == 3:

        # Hidden layer
        input_layer = tf.keras.layers.Input(shape=(input_dimension,))
        # Encoder
        encoder = tf.keras.layers.Dense(int(encoding_dimension),
                                        activation=trial.suggest_categorical('activation_enc_1', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical('kernel_initializer_enc_1',
                                            args.hpo_kernel_initializer)
                                        )(
            input_layer)
        encoder = tf.keras.layers.Dense(int(encoding_dimension / 2),
                                        activation=trial.suggest_categorical('activation_enc_2', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical('kernel_initializer_enc_2',
                                            args.hpo_kernel_initializer)
                                        )(encoder)
        # Decoder
        decoder = tf.keras.layers.Dense(int(encoding_dimension),
                                        activation=trial.suggest_categorical('activation_dec_1', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical('kernel_initializer_dec_1',
                                            args.hpo_kernel_initializer)
                                        )(encoder)
        # Output layer
        decoder = tf.keras.layers.Dense(input_dimension,
                                        activation=trial.suggest_categorical('activation_dec_2', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical('kernel_initializer_dec_2',
                                            args.hpo_kernel_initializer)
                                        )(decoder)

    elif trial.suggest_categorical('hpo_ae_layers', args.hpo_ae_layers) == 5:

        # Hidden layer
        input_layer = tf.keras.layers.Input(shape=(input_dimension,))
        # Encoder
        encoder = tf.keras.layers.Dense(int(encoding_dimension),
                                        activation=trial.suggest_categorical('activation_enc_1', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical(
                                            'kernel_initializer_enc_1', args.hpo_kernel_initializer)
                                        )(input_layer)
        encoder = tf.keras.layers.Dense(int(encoding_dimension / 2),
                                        activation=trial.suggest_categorical('activation_enc_2', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical(
                                            'kernel_initializer_enc_2', args.hpo_kernel_initializer)
                                        )(encoder)
        encoder = tf.keras.layers.Dense(int(encoding_dimension / 4),
                                        activation=trial.suggest_categorical('activation_enc_3', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical(
                                            'kernel_initializer_enc_3', args.hpo_kernel_initializer))(encoder)
        # Decoder
        decoder = tf.keras.layers.Dense(int(encoding_dimension / 2),
                                        activation=trial.suggest_categorical('activation_dec_1', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical(
                                            'kernel_initializer_dec_1', args.hpo_kernel_initializer))(encoder)
        decoder = tf.keras.layers.Dense(int(encoding_dimension),
                                        activation=trial.suggest_categorical('activation_dec_2', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical(
                                            'kernel_initializer_dec_2', args.hpo_kernel_initializer))(decoder)
        # Output layer
        decoder = tf.keras.layers.Dense(input_dimension,
                                        activation=trial.suggest_categorical('activation_dec_3', args.hpo_activation),
                                        kernel_initializer=trial.suggest_categorical(
                                            'kernel_initializer_dec_3', args.hpo_kernel_initializer))(decoder)

    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()
    autoencoder.compile(optimizer=trial.suggest_categorical('optimizer', args.hpo_optimizer), loss='mse')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%sae_trial%s.h5' % (args.checkpoint_dir, trial.number),
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                      mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    autoencoder.fit(x_train, y_train,
                    epochs=args.dnn_num_epochs_auto_encoder,
                    batch_size=args.batch_size_train,
                    shuffle=args.shuffle,  # shuffle instances per epoch
                    validation_split=args.val_model_split,
                    callbacks=[early_stopping, model_checkpoint, lr_reducer]
                    )

    x_pred = autoencoder.predict(x_test)
    mse = sklearn.metrics.mean_squared_error(y_test, x_pred)

    # We perform HPO of the AE removing noise according to the mse
    return mse


def train_nn_wa_classification(args, data_set, label, preprocessor):
    """
    We use an deep artificial neural network for learning the mapping of process instances and labels.
    :param args:
    :param data_set:
    :param label:
    :param preprocessor:
    :return:
    """

    if args.hpo:
        hpo.create_data(data_set, label, preprocessor, args)

        if args.seed:
            sampler = optuna.samplers.TPESampler(seed=args.seed_val)  # Make the sampler behave in a deterministic way.
        else:
            sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(find_best_model, n_trials=args.hpo_eval_runs)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        general.add_to_file(args, "hyper_params", trial)

        return study.best_trial.number
    else:
        train_model(args, data_set, label, preprocessor)
        return -1


def find_best_model(trial):
    args = hpo.args
    x_train = hpo.x_train
    y_train = hpo.y_train
    x_test = hpo.x_test
    y_test = hpo.y_test

    # CNN motivated by the architecture proposed by Abdulrhman et al. (2019).
    input_layer = tf.keras.layers.Input(shape=(hpo.time_steps, hpo.number_attributes),
                                        name='input_layer')

    # Hidden layer one
    layer = tf.keras.layers.Conv1D(filters=trial.suggest_categorical('filters_1', args.hpo_filters),
                                   kernel_size=trial.suggest_categorical('kernel_size_1', args.hpo_kernels_size),
                                   padding=trial.suggest_categorical('padding_1', args.hpo_padding),
                                   activation=trial.suggest_categorical('activation_1', args.hpo_activation),
                                   kernel_initializer=trial.suggest_categorical('kernel_initializer_1',
                                                                                args.hpo_kernel_initializer),
                                   )(input_layer)

    # Hidden layer two
    layer = tf.keras.layers.MaxPool1D()(layer)
    layer = tf.keras.layers.Conv1D(filters=trial.suggest_categorical('filters_2', args.hpo_filters),
                                   kernel_size=trial.suggest_categorical('kernel_size_2', args.hpo_kernels_size),
                                   padding=trial.suggest_categorical('padding_2', args.hpo_padding),
                                   activation=trial.suggest_categorical('activation_2', args.hpo_activation),
                                   kernel_initializer=trial.suggest_categorical('kernel_initializer_2',
                                                                                args.hpo_kernel_initializer),
                                   )(layer)

    if trial.suggest_categorical('conv_layers', args.hpo_conv_layers) == 2:  # Two conv layers
        pass
    elif trial.suggest_categorical('conv_layers', args.hpo_conv_layers) == 3:  # Three conv layers

        # Hidden layer three
        layer = tf.keras.layers.MaxPool1D()(layer)
        layer = tf.keras.layers.Conv1D(filters=trial.suggest_categorical('filters_3', args.hpo_filters),
                                       kernel_size=trial.suggest_categorical('kernel_size_3', args.hpo_kernels_size),
                                       padding=trial.suggest_categorical('padding_3', args.hpo_padding),
                                       activation=trial.suggest_categorical('activation_3', args.hpo_activation),
                                       kernel_initializer=trial.suggest_categorical('kernel_initializer_3',
                                                                                    args.hpo_kernel_initializer),
                                       )(layer)

    elif trial.suggest_categorical('conv_layers', args.hpo_conv_layers) == 4:  # Four conv layers

        # Hidden layer three
        layer = tf.keras.layers.MaxPool1D()(layer)
        layer = tf.keras.layers.Conv1D(filters=trial.suggest_categorical('filters_3', args.hpo_filters),
                                       kernel_size=trial.suggest_categorical('kernel_size_3', args.hpo_kernels_size),
                                       padding=trial.suggest_categorical('padding_3', args.hpo_padding),
                                       activation=trial.suggest_categorical('activation_3', args.hpo_activation),
                                       kernel_initializer=trial.suggest_categorical('kernel_initializer_3',
                                                                                    args.hpo_kernel_initializer),
                                       )(layer)

        # Hidden layer four
        layer = tf.keras.layers.MaxPool1D()(layer)
        layer = tf.keras.layers.Conv1D(filters=trial.suggest_categorical('filters_4', args.hpo_filters),
                                       kernel_size=trial.suggest_categorical('kernel_size_4', args.hpo_kernels_size),
                                       padding=trial.suggest_categorical('padding_4', args.hpo_padding),
                                       activation=trial.suggest_categorical('activation_4', args.hpo_activation),
                                       kernel_initializer=trial.suggest_categorical('kernel_initializer_4',
                                                                                    args.hpo_kernel_initializer),
                                       )(layer)

    elif trial.suggest_categorical('conv_layers', args.hpo_conv_layers) == 5:  # Five conv layers

        # Hidden layer three
        layer = tf.keras.layers.MaxPool1D()(layer)
        layer = tf.keras.layers.Conv1D(filters=trial.suggest_categorical('filters_3', args.hpo_filters),
                                       kernel_size=trial.suggest_categorical('kernel_size_3', args.hpo_kernels_size),
                                       padding=trial.suggest_categorical('padding_3', args.hpo_padding),
                                       activation=trial.suggest_categorical('activation_3', args.hpo_activation),
                                       kernel_initializer=trial.suggest_categorical('kernel_initializer_3',
                                                                                    args.hpo_kernel_initializer),
                                       )(layer)

        # Hidden layer four
        layer = tf.keras.layers.MaxPool1D()(layer)
        layer = tf.keras.layers.Conv1D(filters=trial.suggest_categorical('filters_4', args.hpo_filters),
                                       kernel_size=trial.suggest_categorical('kernel_size_4', args.hpo_kernels_size),
                                       padding=trial.suggest_categorical('padding_4', args.hpo_padding),
                                       activation=trial.suggest_categorical('activation_4', args.hpo_activation),
                                       kernel_initializer=trial.suggest_categorical('kernel_initializer_4',
                                                                                    args.hpo_kernel_initializer),
                                       )(layer)

        # Hidden layer five
        layer = tf.keras.layers.MaxPool1D()(layer)
        layer = tf.keras.layers.Conv1D(filters=trial.suggest_categorical('filters_5', args.hpo_filters),
                                       kernel_size=trial.suggest_categorical('kernel_size_5', args.hpo_kernels_size),
                                       padding=trial.suggest_categorical('padding_5', args.hpo_padding),
                                       activation=trial.suggest_categorical('activation_5', args.hpo_activation),
                                       kernel_initializer=trial.suggest_categorical('kernel_initializer_5',
                                                                                    args.hpo_kernel_initializer),
                                       )(layer)

    # Hidden layer six
    layer = tf.keras.layers.MaxPool1D()(layer)
    layer = tf.keras.layers.Flatten()(layer)

    # Hidden layer seven
    b1 = tf.keras.layers.Dense(trial.suggest_categorical('units_dense', args.hpo_units),
                               kernel_initializer=trial.suggest_categorical('kernel_initializer_6',
                                                                            args.hpo_kernel_initializer),
                               activation=trial.suggest_categorical('activation_dense', args.hpo_activation)
                               )(layer)

    output = tf.keras.layers.Dense(y_train.shape[1],
                                   activation='softmax',
                                   name='output',
                                   kernel_initializer=trial.suggest_categorical(
                                       'kernel_initializer_7', args.hpo_kernel_initializer))(b1)

    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output])

    model.compile(loss={'output': 'categorical_crossentropy'},
                  optimizer=trial.suggest_categorical('optimizer', args.hpo_optimizer),
                  metrics=['accuracy', metric.f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        '%sclf_wa_mapping_trial%s.h5' % (args.checkpoint_dir, trial.number),
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                      mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()
    model.fit(x_train, {'output': y_train},
              validation_split=args.val_model_split,
              verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=args.batch_size_train,
              shuffle=args.shuffle,  # shuffle instances per epoch
              epochs=args.dnn_num_epochs)

    score = model.evaluate(x_test, y_test, verbose=0)

    # We perform HPO of the CNN classifying workarounds according to the f1-score
    return score[2]


def train_model(args, data_set, label, preprocessor):
    """
    Trains a model for activity prediction without HPO (for test purposes during development).
    HPOs are the same as reported in the ECIS2020 Paper.
    """

    number_attributes = preprocessor.data_structure['encoding']['num_values_features'] - 2  # case + time
    time_steps = preprocessor.data_structure['meta']['max_length_process_instance']
    data_set = data_set.reshape((data_set.shape[0], time_steps, number_attributes))

    # CNN according to Abdulrhman et al. (2019)
    input_layer = tf.keras.layers.Input(shape=(time_steps, number_attributes), name='input_layer')
    layer_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(
        input_layer)
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

    output = tf.keras.layers.Dense(label.shape[1], activation='softmax', name='output')(b1)
    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output])

    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(loss={'output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%sclf_wa_mapping.h5' % args.checkpoint_dir,
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                      mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()
    model.fit(data_set, {'output': label},
              validation_split=args.val_model_split,
              verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)
