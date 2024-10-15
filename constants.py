import pathlib

num_epochs = 50000

### Task parameters
task_configs = {
    'lg_transfer_wet_tissue': {
        'dataset_dir' : '/home/shared/LG_Robot/transfer_wet_tissue',
        'num_episodes' : 100,
        'episode_len' : 1000,
        'camera_names' : ['image'],
        'is_sim' : False,
        'state_dim' : 14,
        'no_eval' : True
    },
    'lg_real': {
        'dataset_dir' : '/share0/jellyho/datasets/lg_real',
        'num_episodes' : 60,
        'episode_len' : 1000,
        'camera_names' : ['camera'],
        'is_sim' : False,
        'state_dim' : 14,
        'no_eval' : True
    },
    'lg_sim': {
        'dataset_dir' : '/home/jellyho/LG/datasets/lg_sim',
        'num_episodes' : 100,
        'episode_len' : 500,
        'camera_names' : ['camera'],
        'is_sim' : False,
        'state_dim' : 14,
        'no_eval' : True
    },
    'lg_sim_2': {
        'dataset_dir' : '/share0/jellyho/datasets/lg_sim_2',
        'num_episodes' : 200,
        'episode_len' : 500,
        'camera_names' : ['camera'],
        'is_sim' : False,
        'state_dim' : 14,
        'no_eval' : True
    },
    'transport' : {
        'dataset_dir' : '/share0/jellyho/datasets/transport',
        'num_episodes' : 200,
        'episode_len' : 300,
        'camera_names' : ['camera'],
        'is_sim' : 'robomimic',
        'state_dim' : 14
    },
    'transport_abs' : {
        'dataset_dir' : '/share0/jellyho/datasets/transport_abs',
        'num_episodes' : 200,
        'episode_len' : 300,
        'camera_names' : ['camera'],
        'is_sim' : 'robomimic',
        'state_dim' : 16
    },
    'clean' : {
        'dataset_dir' : '/share0/jellyho/datasets/clean',
        'num_episodes' : 835,
        'episode_len' : 1200,
        'camera_names' : ['angle'],
        'is_sim' : False,
        'state_dim' : 14,
        'no_eval' : True
    }
    
}

model_configs = {
    'chunk30': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 30,
        'hidden_dim' : 512,
        'batch_size' : 256,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : False,
    },
    'chunk30_kl10': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 30,
        'hidden_dim' : 512,
        'batch_size' : 256,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : False,
    },
    'chunk30_kl100': {
        'policy_class' : 'ACT',
        'kl_weight' :  100,
        'chunk_size' : 30,
        'hidden_dim' : 512,
        'batch_size' : 256,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : False,
    },
    'chunk30_kl1': {
        'policy_class' : 'ACT',
        'kl_weight' :  1,
        'chunk_size' : 30,
        'hidden_dim' : 512,
        'batch_size' : 256,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : False,
    },
    'chunk60': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 60,
        'hidden_dim' : 512,
        'batch_size' : 512,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : False,
    },
    'chunk15': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 15,
        'hidden_dim' : 512,
        'batch_size' : 512,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : False,
    },
    'chunk120': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 120,
        'hidden_dim' : 512,
        'batch_size' : 64,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 5e-5,
        'seed' : 0,
        'temporal_agg' : False,
    },
    'chunk5': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 5,
        'hidden_dim' : 512,
        'batch_size' : 512,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : False,
    },
    'agg30': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 30,
        'hidden_dim' : 512,
        'batch_size' : 512,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : True,
    },
    'agg15': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 15,
        'hidden_dim' : 512,
        'batch_size' : 512,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : True,
    },
    'agg5': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 5,
        'hidden_dim' : 512,
        'batch_size' : 512,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : True,
    },
    'agg60': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 60,
        'hidden_dim' : 512,
        'batch_size' : 64,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : True,
    },
    'agg120': {
        'policy_class' : 'ACT',
        'kl_weight' :  10,
        'chunk_size' : 120,
        'hidden_dim' : 512,
        'batch_size' : 64,
        'dim_feedforward' : 3200,
        'num_epochs' : num_epochs,
        'lr' : 1e-4,
        'seed' : 0,
        'temporal_agg' : True,
    }
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2


# import tensorflow as tf 

# DATA_DIR_PATH = 'data'

# FEATURE2DIM = {
#     'info/hz': 1,
#     'steps/action/follow_cube': 8,
#     'steps/action/left_delta_joint': 7,
#     'steps/action/left_delta_pose': 7,
#     'steps/action/left_local_joint': 7,
#     'steps/action/left_local_pose': 7,
#     'steps/action/left_local_pose_quat': 8,
#     'steps/action/right_delta_joint': 7,
#     'steps/action/right_delta_pose': 7,
#     'steps/action/right_local_joint': 7,
#     'steps/action/right_local_pose': 7,
#     'steps/action/right_local_pose_quat': 8,
#     'steps/discount': 1,
#     'steps/is_first': 1,
#     'steps/is_last': 1,
#     'steps/is_terminal': 1,
#     'steps/observation/follow_cube': 8,
#     'steps/observation/image': (480, 640, 3),
#     'steps/observation/left_end_effector_pos': 7,
#     'steps/observation/left_end_effector_pos_quat': 8,
#     'steps/observation/left_grasp_states': 1,
#     'steps/observation/left_joint_states': 7,
#     'steps/observation/left_rexel_command': 7,
#     'steps/observation/left_rexel_joint': 7,
#     'steps/observation/left_rexel_pos': 7,
#     'steps/observation/left_rexel_pos_quat': 8,
#     'steps/observation/natural_language_instruction': 'N/A',
#     'steps/observation/right_end_effector_pos': 7,
#     'steps/observation/right_end_effector_pos_quat': 8,
#     'steps/observation/right_grasp_states': 1,
#     'steps/observation/right_joint_states': 7,
#     'steps/observation/right_rexel_command': 7,
#     'steps/observation/right_rexel_joint': 7,
#     'steps/observation/right_rexel_pos': 7,
#     'steps/observation/right_rexel_pos_quat': 8,
#     'steps/reward': 1   
# }


# FEATURE_DESCRIPTOR = {
#     'steps/is_last': tf.io.VarLenFeature(tf.int64),
#     'steps/observation/left_end_effector_pos_quat': tf.io.VarLenFeature(tf.float32),
#     'steps/observation/right_rexel_joint': tf.io.VarLenFeature(tf.float32),  
#     'steps/observation/left_rexel_command': tf.io.VarLenFeature(tf.float32),  
#     'steps/observation/left_rexel_pos': tf.io.VarLenFeature(tf.float32),  
#     'steps/observation/left_end_effector_pos': tf.io.VarLenFeature(tf.float32),
#     'steps/observation/right_rexel_pos_quat': tf.io.VarLenFeature(tf.float32),
#     'steps/action/right_local_pose': tf.io.VarLenFeature(tf.float32),  
#     'steps/discount': tf.io.VarLenFeature(tf.float32),
#     'steps/action/left_delta_joint': tf.io.VarLenFeature(tf.float32),  
#     'steps/observation/right_end_effector_pos': tf.io.VarLenFeature(tf.float32),
#     'steps/action/follow_cube': tf.io.VarLenFeature(tf.float32),  
#     'steps/observation/image': tf.io.VarLenFeature(tf.string),
#     'steps/action/right_local_joint': tf.io.VarLenFeature(tf.float32),  
#     'steps/action/right_delta_joint': tf.io.VarLenFeature(tf.float32),  
#     'steps/action/right_delta_pose': tf.io.VarLenFeature(tf.float32),  
#     'steps/observation/left_joint_states': tf.io.VarLenFeature(tf.float32),  
#     'steps/action/left_local_pose_quat': tf.io.VarLenFeature(tf.float32),
#     'steps/observation/natural_language_instruction': tf.io.VarLenFeature(tf.string),
#     'steps/observation/right_grasp_states': tf.io.VarLenFeature(tf.int64),  
#     'steps/observation/left_rexel_pos_quat': tf.io.VarLenFeature(tf.float32),
#     'steps/observation/right_joint_states': tf.io.VarLenFeature(tf.float32),  
#     'steps/observation/right_rexel_command': tf.io.VarLenFeature(tf.float32),  
#     'steps/is_first': tf.io.VarLenFeature(tf.int64),
#     'steps/action/left_local_pose': tf.io.VarLenFeature(tf.float32),  
#     'steps/observation/right_end_effector_pos_quat': tf.io.VarLenFeature(tf.float32),
#     'steps/observation/left_rexel_joint': tf.io.VarLenFeature(tf.float32),  
#     'steps/observation/right_rexel_pos': tf.io.VarLenFeature(tf.float32),  
#     'steps/observation/follow_cube': tf.io.VarLenFeature(tf.float32),  
#     'steps/reward': tf.io.VarLenFeature(tf.float32),
#     'steps/action/right_local_pose_quat': tf.io.VarLenFeature(tf.float32),
#     'steps/is_terminal': tf.io.VarLenFeature(tf.int64),
#     'steps/observation/left_grasp_states': tf.io.VarLenFeature(tf.int64),  
#     'steps/action/left_local_joint': tf.io.VarLenFeature(tf.float32),  
#     'steps/action/left_delta_pose': tf.io.VarLenFeature(tf.float32),  
#     'info/hz': tf.io.VarLenFeature(tf.float32)
# }