from robomimic.envs.env_base import EnvBase
import json
import os
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from deoxys_vision.networking.camera_redis_interface import \
    CameraRedisSubInterface
from deoxys_vision.utils.camera_utils import get_camera_info
from deoxys.experimental.motion_utils import reset_joints_to
import robomimic.utils.obs_utils as ObsUtils


from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils as T

from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.utils.input_utils import input2action
# from deoxys.k4a_interface import K4aInterface
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
import cv2


def zed_image_transform_fn(image):
    assert len(image.shape) == 3, "expected image to have 3 dimensions, got {}".format(len(image.shape))
    height, width = image.shape[:2]
    new_width = 720
    x_start = (width - new_width) // 2
    image = image[:, x_start:x_start + new_width]
    image = cv2.resize(image, (128,128)).astype(np.uint8)
    return image

def rs_image_transform_fn(image):
    assert len(image.shape) == 3, "expected image to have 3 dimensions, got {}".format(len(image.shape))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (128,128)).astype(np.uint8)
    return image

def get_eef_pos(eef_state):
    O_T_EE = np.array(eef_state).reshape(4, 4).transpose()
    return O_T_EE[:3, 3]

def get_eef_quat(eef_state):
    O_T_EE = np.array(eef_state).reshape(4, 4).transpose()
    return T.mat2quat(O_T_EE[:3, :3])

def get_gripper_pos(gripper_state):
    return gripper_state[..., np.newaxis]

CONVERSION_MAP = {
    "obs/robot0_agentview_right_image": {
        "filename": "testing_demo_camera_zed_0.npz",
        "transform": zed_image_transform_fn,
        "key": "camera_zed_0"
    },
    "obs/robot0_agentview_left_image": {
        "filename": "testing_demo_camera_zed_1.npz",
        "transform": zed_image_transform_fn,
        "key": "camera_zed_1"
    },
    "obs/robot0_eye_in_hand_image": {
        "filename": "testing_demo_camera_rs_0.npz",
        "transform": rs_image_transform_fn,
        "key": "camera_rs_0"
    },
    "obs/robot0_eef_pos": {
        "filename": "testing_demo_ee_states.npz",
        "transform": get_eef_pos,
        "key": "ee_states"
    },
    "obs/robot0_eef_quat": {
        "filename": "testing_demo_ee_states.npz",
        "transform": get_eef_quat,
        "key": "ee_states"
    },
    "obs/robot0_gripper_qpos": {
        "filename": "testing_demo_gripper_states.npz",
        "transform": get_gripper_pos,
        "key": "gripper_states"
    },
    "obs/robot0_joint_pos": {
        "filename": "testing_demo_joint_states.npz",
        "transform": lambda x: x,
        "key": "joint_states"
    },
    "actions": {
        "filename": "testing_demo_action.npz",
        "transform": lambda x: x,
        "key": "action"
    }
}


class EnvRealDeoxys(EnvBase):
    """
    Real robot environment for Deoxys controller. This environment is used for collecting
    data on the real robot.

     env = env_class(
        env_name=env_name, 
        render=render, 
        render_offscreen=render_offscreen, 
        use_image_obs=use_image_obs,
        postprocess_visual_obs=True,
        env_lang=env_lang,
        **kwargs,
    )
    """

    def __init__(
            self, 
            env_name="real_deoxys", 
            vendor_id=9583, 
            product_id=50741, 
            interface_cfg="/home/soroush/code/deoxys_control/deoxys/config/bulbasaur.yml", 
            controller_cfg = "/home/soroush/code/deoxys_control/deoxys/config/osc-pose-controller.yml",
            controller_type="OSC_POSE",
            camera_ids=["rs_0", "zed_0", "zed_1"], 
            postprocess_visual_obs=True,
            render=False,
            render_offscreen=False,
            use_image_obs=True,
            env_lang=None,
            use_object_obs=False,
        ):
        """
        Args:
            env_name (str): name of the environment
            env_params (dict): dictionary of environment-specific parameters
            robot_params (dict): dictionary of robot-specific parameters
        """
        self._name = env_name
        # self.device = SpaceMouse(vendor_id=vendor_id, product_id=product_id)
        self.interface_cfg = interface_cfg
        self.robot_interface = FrankaInterface(interface_cfg)
        self.camera_ids = camera_ids
        self.cr_interfaces = {}
        for camera_id in camera_ids:
            cr_interface = CameraRedisSubInterface(camera_info=get_camera_info(camera_id), redis_host="127.0.0.1")
            cr_interface.start()
            self.cr_interfaces[camera_id] = cr_interface
        self.controller_cfg = YamlConfig(controller_cfg).as_easydict()
        self.controller_type = controller_type
        self.postprocess_visual_obs = postprocess_visual_obs
    
    def step(self, action):
        if not( np.all(action <= 1) and np.all(action >= -1)):
            action = np.clip(action, -1, 1)

        self.robot_interface.control(
            controller_type=self.controller_type,
            action=action,
            controller_cfg=self.controller_cfg,
        )
        obs = self.get_observation()
        r = 0
        info = {}
        return obs, r, self.is_done(), info

    def is_success(self):
        return  {"task": False} 
    
    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        return False 
    

    def close(self):
        """
        Close the environment.
        """
        reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ] 
        reset_joints_to(self.robot_interface, np.array(reset_joint_positions), gripper_open=True)
        self.robot_interface.close()

    def reset(self):
        """
        Reset the environment.
        """
        reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ] 
        reset_joints_to(self.robot_interface, np.array(reset_joint_positions), gripper_open=True)
        # might not be needed???
        while len(self.robot_interface._state_buffer) == 0:
            continue
        return self.get_observation()

    def get_observation(self, obs=None):
        """
        Get the current observation from the environment.

        Returns:
            dict: dictionary containing the current observation
        """
        raw_data = {}
        last_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        raw_data = {
            "ee_states": np.array(last_state.O_T_EE),
            "joint_states": np.array(last_state.q),
            "gripper_states": np.array(last_gripper_state.width),
        }

        for camera_id in self.camera_ids:
            img = self.cr_interfaces[camera_id].get_img()
            raw_data[f"camera_{camera_id}"] = img["color"]

        obs = {}
        for key, data in CONVERSION_MAP.items():
            if "obs" in key:
                obs_key = key.split("/")[1]
                obs[obs_key] = data["transform"](raw_data[data["key"]])

        ret = {}
        for k in obs:
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                # ret[k] = obs[k][::-1]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=obs[k], obs_key=k)
            else:
                ret[k] = obs[k]

        return ret
    
    def render(self, mode="human", height=None, width=None, camera_name=None):
        image = self.cr_interfaces[camera_name].get_img()["color"]
        _, w = image.shape[:2]
        new_width = 720
        x_start = (w - new_width) // 2
        image = image[:, x_start:x_start + new_width]
        image = cv2.resize(image, (height,width)).astype(np.uint8)
        return image
    
    def get_goal(self):
        raise NotImplementedError("Goal-based tasks are not supported for this environment.")
    
    def get_reward(self):
        return 0
    
    def get_state(self):
        raise NotImplementedError("State-based tasks are not supported for this environment.")
    
    def reset_to(self, state):
        raise NotImplementedError("Resetting to a specific state is not supported for this environment.")
    
    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return (Exception)


    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return 7
    
    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._name
    
    def serialize(self):
        raise NotImplementedError("Serializing this environment is not supported.")
    
    def set_goal(self, **kwargs):
        raise NotImplementedError("Setting goals is not supported for this environment.")
    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return 4
    
    def create_for_data_processing(cls, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        raise NotImplementedError("Creating environment for data processing is not supported for this environment.")