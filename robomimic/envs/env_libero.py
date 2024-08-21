"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import os
import numpy as np
from copy import deepcopy

import robosuite
try:
    import robocasa
except ImportError:
    pass

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils
import robomimic.envs.env_base as EB
from robomimic.macros import LANG_EMB_KEY
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T


class EnvLibero(EB.EnvBase):
    """Wrapper class for libero environments"""
    def __init__(
        self, 
        env_name, 
        env_meta,
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True,
        env_lang=None, 
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).

            lang: TODO add documentation
        """
        self.postprocess_visual_obs = postprocess_visual_obs

        self._env_name = env_name
        task_bddl_file = os.path.join(f"/mmfs1/gscratch/weirdlab/jacob33/retrieval/LIBERO/{env_meta['bddl_file_name']}", )
                
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128
        }
        self.env = OffScreenRenderEnv(**env_args)
        
        self.env_lang = env_lang

        self.env_meta = env_meta
        
        # Make sure joint position observations and eef vel observations are active
        for ob_name in self.env.env.observation_names:
            if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                self.env.env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)
        
        

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        info["is_success"] = self.is_success()
        return obs, r, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        
        #TODO: this does not reset to one of the "allowed" reset poses in the benchmark
        di = self.env.reset()
        
        # keep track of episode language and embedding
        if self.env_lang is not None:
            self._ep_lang_str = self.env_lang
        else:
            self._ep_lang_str = "dummy"

        # self._ep_lang_emb = LangUtils.get_lang_emb(self._ep_lang_str)
        
        return self.get_observation(di)
    
    #notifies the environment whether or not the next environemnt testing object should update its category
    def update_env(self, attr, value):
        setattr(self.env, attr, value)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        raise NotImplementedError("reset_to not implemented for Libero environments")
        should_ret = False
        if "model" in state:
            if state.get("ep_meta", None) is not None:
                # set relevant episode information
                ep_meta = json.loads(state["ep_meta"])
            else:
                ep_meta = {}
            if hasattr(self.env, "set_attrs_from_ep_meta"): # older versions had this function
                self.env.set_attrs_from_ep_meta(ep_meta)
            elif hasattr(self.env, "set_ep_meta"): # newer versions
                self.env.set_ep_meta(ep_meta)
            # this reset is necessary.
            # while the call to env.reset_from_xml_string does call reset,
            # that is only a "soft" reset that doesn't actually reload the model.
            self.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml
                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(state["model"])

            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
            if hasattr(self.env, "unset_ep_meta"): # unset the ep meta after reset complete
                self.env.unset_ep_meta()
        if "states" in state:
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True

        # update state as needed
        if hasattr(self.env, "update_sites"):
            # older versions of environment had update_sites function
            self.env.update_sites()
        if hasattr(self.env, "update_state"):
            # later versions renamed this to update_state
            self.env.update_state()

        if "goal" in state:
            self.set_goal(**state["goal"])
        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            return self.get_observation()
        return None

    def render(self, mode="human", height=None, width=None, camera_name=None):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        # if camera_name is None, infer from initial env kwargs
        if camera_name is None:
            camera_name = "agentview"

        if mode == "human":
            raise not NotImplementedError("human rendering not implemented for Libero environments")
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif mode == "rgb_array":
            return self.env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        assert di is not None, "di must be provided to get_observation"
        # The keys are wrong so we need to fix them
        new_di = {}
        new_di["ee_pos"] = di["robot0_eef_pos"]
        new_di["ee_ori"] = T.quat2axisangle(di["robot0_eef_quat"])
        new_di["ee_states"] = np.hstack((di["robot0_eef_pos"],  new_di["ee_ori"]))
        new_di["joint_states"] = di["robot0_joint_pos"]
        new_di["gripper_states"] = di["robot0_gripper_qpos"]
        
        new_di["agentview_rgb"] = ObsUtils.process_obs(obs=di["agentview_image"], obs_key='agentview_rgb')
        new_di["eye_in_hand_rgb"] = ObsUtils.process_obs(obs=di["robot0_eye_in_hand_image"], obs_key='eye_in_hand_rgb')
        
        # add in eef pose to not break other code that has it hardcoded:
        new_di["robot0_eef_pos"] = di["robot0_eef_pos"]     
        di = new_di
        # ret = {}
        # for k in di:
        #     if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
        #         ret[k] = di[k][::-1]
        #         if self.postprocess_visual_obs:
        #             ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

        # # "object" key contains object information
        # if "object-state" in di:
        #     ret["object"] = np.array(di["object-state"])

        # for robot in self.env.robots:
        #     # add all robot-arm-specific observations. Note the (k not in ret) check
        #     # ensures that we don't accidentally add robot wrist images a second time
        #     pf = robot.robot_model.naming_prefix
        #     for k in di:
        #         if k.startswith(pf) and (k not in ret) and \
        #                 (not k.endswith("proprio-state")):
        #             ret[k] = np.array(di[k])


        # ret["lang_emb"] = np.array(self._ep_lang_emb)
        return new_di

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        info = dict(model=xml, states=state)
        if hasattr(self.env, "get_ep_meta"):
            # get ep_meta if applicable
            info["ep_meta"] = json.dumps(self.env.get_ep_meta(), indent=4)
        return info

    def get_reward(self):
        """
        Get current reward.
        """
        return self.env.env.reward()

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError("get_goal not implemented for Robosuite environments")
        return self.get_observation(self.env._get_goal())

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError("set_goal not implemented for Robosuite environments")
        return self.env.set_goal(**kwargs)

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.env.action_spec[0].shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.ROBOSUITE_TYPE

    @property
    def version(self):
        """
        Returns version of robosuite used for this environment, eg. 1.2.0
        """
        return robosuite.__version__

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @classmethod
    def create_for_data_processing(
        cls, 
        env_name, 
        camera_names, 
        camera_height, 
        camera_width, 
        reward_shaping, 
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
        """
        raise NotImplementedError("create_for_data_processing not implemented for Libero environments")
        is_v1 = (robosuite.__version__.split(".")[0] == "1")
        has_camera = (len(camera_names) > 0)

        new_kwargs = {
            "reward_shaping": reward_shaping,
        }

        if has_camera:
            if is_v1:
                new_kwargs["camera_names"] = list(camera_names)
                new_kwargs["camera_heights"] = camera_height
                new_kwargs["camera_widths"] = camera_width
            else:
                assert len(camera_names) == 1
                if has_camera:
                    new_kwargs["camera_name"] = camera_names[0]
                    new_kwargs["camera_height"] = camera_height
                    new_kwargs["camera_width"] = camera_width

        kwargs.update(new_kwargs)

        # also initialize obs utils so it knows which modalities are image modalities
        image_modalities = list(camera_names)
        if is_v1:
            image_modalities = ["{}_image".format(cn) for cn in camera_names]
        elif has_camera:
            # v0.3 only had support for one image, and it was named "rgb"
            assert len(image_modalities) == 1
            image_modalities = ["rgb"]
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False, 
            render_offscreen=has_camera, 
            use_image_obs=has_camera, 
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return (Exception)

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name # + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
