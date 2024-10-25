"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner


SHAPE_META = {
    "obs": {
        "robot0_agentview_left_image": {
            "shape": [3, 128, 128],
            "type": "rgb"
        },
        "robot0_eye_in_hand_image": {
            "shape": [3, 128, 128],
            "type": "rgb"
        },
        "robot0_eef_pos": {
            "shape": [3]
        },
        "robot0_eef_quat": {
            "shape": [4]
        },
        "robot0_gripper_qpos": {
            "shape": [1]
        }
    },
    "action": {
        "shape": [7]
    }
}

DEOXYS_ENV_INFO = {
    "env_name": "EnvRealDeoxys",
    "type": 4,
    "env_kwargs": {}
}

ENV_RUNNER_KWARGS = {
    "shape_meta": SHAPE_META,
    "dataset_path": None,
    "n_train": 0,
    "n_train_vis": 0,
    "n_test": 1,
    "n_test_vis": 1,
    "render_obs_key": "robot0_agentview_left_image",
    "env_meta": DEOXYS_ENV_INFO,
}

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    ENV_RUNNER_KWARGS["output_dir"] = output_dir

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = RobomimicImageRunner(**ENV_RUNNER_KWARGS)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
