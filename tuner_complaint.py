from kerastuner.engine import base_tuner
import kerastuner as kt
from numpy import dtype
from sympy import factor
from tensorflow import keras
from tensorflow.keras import layers

from typing import NamedTuple, Dict, Text, Any
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

LABEL_KEY = "Product"
FEATURE_KEY = "Consumer_complaint"

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100
embedding_dim = 16
num_classes = 4

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

LABEL_KEY = "Product"
FEATURE_KEY = "Consumer_complaint"

def transformed_name(key):
    key = key.replace('-', '_')
    return key + '_xf'

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                            ('fit_kwargs', Dict[Text,Any])])



def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern,
             tf_transform_output,
             num_epochs=10,
             batch_size=64)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transformed_name(LABEL_KEY))
    return dataset

def model_builder(hp):
    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    # Hyperparameter tuning for the number of hidden layers
    num_hidden_layers = hp.Int('hidden_layers', min_value=1, max_value=5, default=3)

    # Dynamically add hidden layers based on the chosen hyperparameter
    for i in range(num_hidden_layers):
        x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=['accuracy']
    )

    model.summary()
    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='kt_random_search'
    )

    # load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = _input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = _input_fn(fn_args.eval_files, tf_transform_output, 10)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10)

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)]])


    return TunerFnResult(
        tuner = tuner,
        fit_kwargs={
            "callbacks":[stop_early],
            "x": train_set,
            "validation_data": val_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 10
        }
    )
