import tensorflow as tf 

FEATURE2DIM_LG_SIM = {
    'steps/discount': 1,
    'steps/is_first': 1,
    'steps/is_last': 1,
    'steps/is_terminal': 1,
    'steps/reward': 1,
    'steps/action/delta_joint': 14,
    'steps/action/delta_pose': 16,
    'steps/action/local_joint': 14,
    'steps/action/local_pose': 16,
    'steps/observation/image': (480, 640, 3),
    'steps/observation/end_effector_pos': 16,
    'steps/observation/joint_pos': 14,
    'steps/observation/natural_language_instruction': 1,
}
FEATURE_DESCRIPTOR_LG_SIM = {
    'steps/is_last': tf.io.VarLenFeature(tf.int64),
    'steps/reward': tf.io.VarLenFeature(tf.float32),
    'steps/is_terminal': tf.io.VarLenFeature(tf.int64),
    'steps/is_first': tf.io.VarLenFeature(tf.int64),
    'steps/discount': tf.io.VarLenFeature(tf.float32),
    'steps/observation/image': tf.io.VarLenFeature(tf.string),
    'steps/observation/end_effector_pos': tf.io.VarLenFeature(tf.float32), 
    'steps/observation/joint_pos': tf.io.VarLenFeature(tf.float32),
    'steps/action/local_joint': tf.io.VarLenFeature(tf.float32),  
    'steps/action/local_pose': tf.io.VarLenFeature(tf.float32),  
    'steps/action/delta_joint': tf.io.VarLenFeature(tf.float32),  
    'steps/action/delta_pose': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/natural_language_instruction': tf.io.VarLenFeature(tf.string),
}


FEATURE2DIM_LG_REAL = {
    'info/hz': 1,
    'steps/action/follow_cube': 8,
    'steps/action/left_delta_joint': 7,
    'steps/action/left_delta_pose': 7,
    'steps/action/left_local_joint': 7,
    'steps/action/left_local_pose': 7,
    'steps/action/left_local_pose_quat': 8,
    'steps/action/right_delta_joint': 7,
    'steps/action/right_delta_pose': 7,
    'steps/action/right_local_joint': 7,
    'steps/action/right_local_pose': 7,
    'steps/action/right_local_pose_quat': 8,
    'steps/discount': 1,
    'steps/is_first': 1,
    'steps/is_last': 1,
    'steps/is_terminal': 1,
    'steps/observation/follow_cube': 8,
    'steps/observation/image': (480, 640, 3),
    'steps/observation/left_end_effector_pos': 7,
    'steps/observation/left_end_effector_pos_quat': 8,
    'steps/observation/left_grasp_states': 1,
    'steps/observation/left_joint_states': 7,
    'steps/observation/left_rexel_command': 7,
    'steps/observation/left_rexel_joint': 7,
    'steps/observation/left_rexel_pos': 7,
    'steps/observation/left_rexel_pos_quat': 8,
    'steps/observation/natural_language_instruction': 1,
    'steps/observation/right_end_effector_pos': 7,
    'steps/observation/right_end_effector_pos_quat': 8,
    'steps/observation/right_grasp_states': 1,
    'steps/observation/right_joint_states': 7,
    'steps/observation/right_rexel_command': 7,
    'steps/observation/right_rexel_joint': 7,
    'steps/observation/right_rexel_pos': 7,
    'steps/observation/right_rexel_pos_quat': 8,
    'steps/reward': 1   
}
FEATURE_DESCRIPTOR_LG_REAL = {
    'steps/is_last': tf.io.VarLenFeature(tf.int64),
    'steps/observation/left_end_effector_pos_quat': tf.io.VarLenFeature(tf.float32),
    'steps/observation/right_rexel_joint': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/left_rexel_command': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/left_rexel_pos': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/left_end_effector_pos': tf.io.VarLenFeature(tf.float32),
    'steps/observation/right_rexel_pos_quat': tf.io.VarLenFeature(tf.float32),
    'steps/action/right_local_pose': tf.io.VarLenFeature(tf.float32),  
    'steps/discount': tf.io.VarLenFeature(tf.float32),
    'steps/action/left_delta_joint': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/right_end_effector_pos': tf.io.VarLenFeature(tf.float32),
    'steps/action/follow_cube': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/image': tf.io.VarLenFeature(tf.string),
    'steps/action/right_local_joint': tf.io.VarLenFeature(tf.float32),  
    'steps/action/right_delta_joint': tf.io.VarLenFeature(tf.float32),  
    'steps/action/right_delta_pose': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/left_joint_states': tf.io.VarLenFeature(tf.float32),  
    'steps/action/left_local_pose_quat': tf.io.VarLenFeature(tf.float32),
    'steps/observation/natural_language_instruction': tf.io.VarLenFeature(tf.string),
    'steps/observation/right_grasp_states': tf.io.VarLenFeature(tf.int64),  
    'steps/observation/left_rexel_pos_quat': tf.io.VarLenFeature(tf.float32),
    'steps/observation/right_joint_states': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/right_rexel_command': tf.io.VarLenFeature(tf.float32),  
    'steps/is_first': tf.io.VarLenFeature(tf.int64),
    'steps/action/left_local_pose': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/right_end_effector_pos_quat': tf.io.VarLenFeature(tf.float32),
    'steps/observation/left_rexel_joint': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/right_rexel_pos': tf.io.VarLenFeature(tf.float32),  
    'steps/observation/follow_cube': tf.io.VarLenFeature(tf.float32),  
    'steps/reward': tf.io.VarLenFeature(tf.float32),
    'steps/action/right_local_pose_quat': tf.io.VarLenFeature(tf.float32),
    'steps/is_terminal': tf.io.VarLenFeature(tf.int64),
    'steps/observation/left_grasp_states': tf.io.VarLenFeature(tf.int64),  
    'steps/action/left_local_joint': tf.io.VarLenFeature(tf.float32),  
    'steps/action/left_delta_pose': tf.io.VarLenFeature(tf.float32),  
    'info/hz': tf.io.VarLenFeature(tf.float32)
}