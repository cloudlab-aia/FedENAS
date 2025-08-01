import os
import sys
import pickle as pickle
import numpy as np


def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print(file_name)
    with open(os.path.join(data_path, file_name), 'rb') as finp:
      data = pickle.load(finp, encoding="latin1")
      images.append(data["data"].astype(np.float32) / 255.0)
      labels.append(np.array(data["labels"], dtype=np.int32))
  return (
    np.transpose(np.reshape(np.concatenate(images, axis=0), [-1, 3, 32, 32]), [0, 2, 3, 1]),
    np.concatenate(labels, axis=0))


def read_data(data_path, num_valids=5000):
  print("-" * 80)
  print("Reading data")

  images, labels = {}, {}
  images["train"], labels["train"] = _read_data(data_path, [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ])

  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  images["test"], labels["test"] = _read_data(data_path, [
    "test_batch",
  ])

  print("Prepropcess: [subtract mean], [divide std]")
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
  print("std: {}".format(np.reshape(std * 255.0, [-1])))

  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  # with open('/workspace/Proof-of-concept/data/cifar10/all_data.pkl', 'wb') as f:
  #     pickle.dump({"images":images,"labels":labels}, f)
  
  return images, labels

