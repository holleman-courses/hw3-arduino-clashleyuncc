import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Input, layers

sinModel = tf.keras.models.load_model('workspace/src/sin_predictor.h5')
sinModel.summary()

num_train_examples = 20000
sequence_length = 8
batch_size = 64
num_epochs = 25
val_split = 0.2
rng = np.random.default_rng(2024)
frequencies = rng.uniform(0.02, 0.2, size=num_train_examples)
phase_offsets = rng.uniform(0.0, 2*np.pi, size=num_train_examples)
sequences = np.zeros((num_train_examples, sequence_length))
# Generate sine waves
for i in range(num_train_examples):
    sequences[i] = np.sin(2*np.pi*frequencies[i]* np.arange(sequence_length) + phase_offsets[i])
x_train = sequences[:, :sequence_length-1]

num_calibration_steps = 25
#converter = tf.lite.TFLiteConverter.from_saved_model('sin_predictor.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(sinModel)

if True:
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  def representative_dataset_gen():
    for i in range(num_calibration_steps):
      next_input = x_train[i:i+1,:]
      yield [next_input.astype(np.float32)] ## yield defines a generator

  converter.representative_dataset = representative_dataset_gen
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

  converter.inference_input_type = tf.int8  # or tf.uint8; should match dat_q in eval_quantized_model.py
  converter.inference_output_type = tf.int8  # or tf.uint8

tflite_quant_model = converter.convert()

tfl_file_name = 'sinModel2.tflite'

with open(tfl_file_name, "wb") as fpo:
  fpo.write(tflite_quant_model)
print(f"Wrote to {tfl_file_name}")


#!ls -l $tfl_file_name

# xdd -i sinModel2.tflite