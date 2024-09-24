from models import *

test_horses, test_zebras = dataset['testA'], dataset['testB'] 

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

TEST_SIZE = 5

for inp in test_horses.take(TEST_SIZE):
  generate_images(generator_g, inp)