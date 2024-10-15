### TFRecord utilities

import tensorflow as tf
from typing import Dict
from glob import glob 
import os 
import numpy as np

import h5py
import cv2
from tqdm import tqdm

def get_dataset_from_tfrecord(data_dir_path: str): 
    tfrecord_files = glob(os.path.join(data_dir_path, '*.tfrecord-*'))
    print(tfrecord_files)
    raw_dataset = tf.data.TFRecordDataset(filenames=tfrecord_files)
    return raw_dataset

def get_parsed_dataset(dataset: tf.data.TFRecordDataset, feature_description: Dict, feature2dim: Dict):
    datasets = []
    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)
    parsed_dataset = dataset.map(_parse_image_function)
    for record in parsed_dataset:
        dataset_dict = {} 
        # this runs only once because the dataset is a single record
        for key, value in record.items(): 
            value = tf.sparse.to_dense(value).numpy()
            if feature2dim[key] == 7 or feature2dim[key] == 8 or feature2dim[key] == 14 or feature2dim[key] == 16: 
                value = value.reshape(-1, feature2dim[key]) 
            dataset_dict[key] = value
        datasets.append(dataset_dict)
    return datasets

def decode_images(images):
    imgs = []
    print('decoding images')
    for img in tqdm(images): 
        imgs.append(cv2.imdecode(np.fromstring(img, dtype=np.uint8), cv2.IMREAD_COLOR))
    return np.stack(imgs, axis=0)

def tfrecords_to_hdf5_sim(tf_dir, hd_dir, descriptor, f2dim, prefix=0):
    if not os.path.exists(hd_dir):
        os.makedirs(hd_dir)

    tfrecord_files = glob(os.path.join(tf_dir, '*.tfrecord-*'))
    eps = prefix
    for i, tf_file_dir in enumerate(tfrecord_files):
        raw_dataset = tf.data.TFRecordDataset(tf_file_dir)
        datasets = get_parsed_dataset(raw_dataset, descriptor, feature2dim=f2dim)
        print(f'{tf_file_dir} has {len(datasets)} demos')
        for dataset in datasets:
            # print(list(dataset.keys()))
            max_timesteps = len(dataset['steps/observation/image'])
            
            image_decoded = decode_images(dataset['steps/observation/image'])
            # print(image_decoded.shape, max_timesteps)

            data_dict = {
                '/observations/qpos': [],
                '/action': [],
                '/observations/images/camera': [],
                '/observations/natural_language_instruction' : []
            }

            data_dict['/observations/images/camera'] = image_decoded
            data_dict['/observations/qpos'] = dataset['steps/observation/joint_pos']
            data_dict['/action'] = dataset['steps/action/local_joint']
            data_dict['/observations/natural_language_instruction'] = dataset['steps/observation/natural_language_instruction']

            # print(data_dict['/observations/natural_language_instruction'])

            with h5py.File(f'{hd_dir}/episode_{eps}.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = False
                obs = root.create_group('observations')
                image = obs.create_group('images')
                image.create_dataset('camera', (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3), )
                qpos = obs.create_dataset('qpos', (max_timesteps, 14))
                action = root.create_dataset('action', (max_timesteps, 14))
                lang = obs.create_dataset('natural_language_instruction', (max_timesteps, ), dtype=h5py.string_dtype(encoding='utf-8'))

                for name, array in data_dict.items():
                    root[name][...] = array

            print(f'Converted to {hd_dir}/episode_{eps}.hdf5')
            eps += 1

def tfrecords_to_hdf5_real(tf_dir, hd_dir, descriptor, f2dim):
    if not os.path.exists(hd_dir):
        os.makedirs(hd_dir)

    tfrecord_files = glob(os.path.join(tf_dir, '*.tfrecord-*'))
    eps = 0
    for i, tf_file_dir in enumerate(tfrecord_files):
        raw_dataset = tf.data.TFRecordDataset(tf_file_dir)
        datasets = get_parsed_dataset(raw_dataset, descriptor, feature2dim=f2dim)
        print(f'{tf_file_dir} has {len(datasets)} demos')
        for dataset in datasets:
            max_timesteps = len(dataset['steps/observation/image'])
            
            image_decoded = decode_images(dataset['steps/observation/image'])
            data_dict = {
                '/observations/qpos': [],
                '/action': [],
                '/observations/images/camera': [],
            }

            data_dict['/observations/images/camera'] = image_decoded
            data_dict['/observations/qpos'] = np.concatenate((dataset['steps/observation/left_joint_states'], dataset['steps/observation/right_joint_states']), axis=1)
            data_dict['/action'] = np.concatenate((dataset['steps/action/left_local_joint'], dataset['steps/action/right_local_joint']), axis=1)

            with h5py.File(f'{hd_dir}/episode_{eps}.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = False
                obs = root.create_group('observations')
                image = obs.create_group('images')
                image.create_dataset('camera', (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3), )
                qpos = obs.create_dataset('qpos', (max_timesteps, 14))
                action = root.create_dataset('action', (max_timesteps, 14))

                for name, array in data_dict.items():
                    root[name][...] = array

            print(f'Converted to {hd_dir}/episode_{eps}.hdf5')
            eps += 1


### Unit test
if __name__ == "__main__": 
    from dataset_const import FEATURE_DESCRIPTOR_LG_REAL, FEATURE2DIM_LG_REAL
    from dataset_const import FEATURE_DESCRIPTOR_LG_SIM, FEATURE2DIM_LG_SIM

    tfrecords_to_hdf5_sim('/home/jellyho/sim_ann_test', '/home/jellyho/datasets/sim_ann_test', FEATURE_DESCRIPTOR_LG_SIM, FEATURE2DIM_LG_SIM, prefix=0)
    
    # with h5py.File('robosuite_handover/demo.hdf5', 'r') as root:
    #     print(root['/data'])
    #     for key in root.keys():
    #         print(key)

    # raw_dataset = get_dataset_from_tfrecord(DATA_DIR_PATH)
    # assert isinstance(raw_dataset, tf.data.TFRecordDataset)
    # print("Raw dataset loaded successfully.")
    # dataset = get_parsed_dataset(raw_dataset, FEATURE_DESCRIPTOR, feature2dim=FEATURE2DIM)
    # print(dataset['steps/observation/left_joint_states'][0:2])
    # print(dataset['steps/observation/left_rexel_command'][0:2])
    # # print(dataset.keys())
    # # print(dataset['info/hz'])
    # assert len(dataset.keys()) == 36, "Please check the data format."
    # print("Parse data successfully.")
    # # print(dataset.keys())
    
    # print("All tests passed.")
