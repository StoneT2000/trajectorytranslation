import pickle
import gym
from omegaconf import OmegaConf
import pandas as pd

from tr2.models.translation.lstm import LSTM
from tr2.models.translation.mlp_id import MLPTranslationID, MLPTranslationIDTeacherStudentActorCritic
import torch
from tr2.models.translation.translation_transformer import (
    TranslationTeacherStudentActorCritic,
    TranslationTransformerGPT2,
)
from tr2.models.translation.convnet import TranslationConvNet

from tr2.utils.animate import animate
import os.path as osp
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from tr2.data.teacherstudent import TeacherStudentDataset
from tr2.data.utils import MinMaxScaler
from paper_rl.common.rollout import Rollout
from paper_rl.cfg import parse

torch.manual_seed(1)

def main(cfg):
    ckpt = torch.load(cfg.model, map_location=cfg.device)
    model_cfg = ckpt["cfg"]["model_cfg"]
    print(f"=== Evaluating {cfg.model} on {cfg.test_n} trajectories; Epoch: {ckpt['epoch']} ===")
    env_cfg = cfg.env_cfg
    env_cfg.stack_size = model_cfg["stack_size"]
    env_cfg.max_trajectory_length = model_cfg["max_teacher_length"]
    env_cfg.trajectory_sample_skip_steps = model_cfg["trajectory_sample_skip_steps"]

    device = torch.device(cfg.device)
    model_cls = TranslationTransformerGPT2
    if model_cfg["type"].lower() == "lstm":
        model_cls = LSTM
    elif model_cfg["type"] == "ConvNet":
        model_cls = TranslationConvNet
    elif model_cfg["type"] == "MLPID":
        model_cls = MLPTranslationID
    model = model_cls.load_from_checkpoint(ckpt, device=device)
    save_video=cfg.save_video
    use_teacher=not cfg.ignore_teacher

    device = torch.device(cfg.device)
    model.eval()

    ids = list(np.load(cfg.env_cfg.trajectories))[:]
    np.random.seed(0)
    id_rng = np.random.default_rng(1)
    id_rng.shuffle(ids)
    ids = ids[:cfg.test_n]
    n_envs = cfg.n_envs
    rollout = Rollout()
    def obs_to_tensor(o):
        tensor_o = {}
        for k in o:
            tensor_o[k] = torch.as_tensor(o[k], device=device)
            if tensor_o[k].dtype == torch.float64:
                tensor_o[k] = tensor_o[k].float()
        return tensor_o
    

    solved_trajectories = dict(teacher=dict(), student=dict())
    def format_trajectory(t_obs, t_act, t_rew, t_info):
        ep_len = len(t_act)
        t_info = t_info[-1]
        data = dict(
            returns=np.sum(t_rew),
            traj_match=t_info["stats"]["farthest_traj_match_frac"],
            match_left=t_info["traj_len"] - t_info["stats"]["farthest_traj_match"] - 1,
            traj_len=t_info["traj_len"],
            ep_len=ep_len,
            success=t_info['task_complete'],
            traj_id=t_info["traj_id"],
            plans=t_info["stats"]["plans"]
        )
        if cfg.save_solved_trajectories and data["success"]:
            if env_cfg["stack_size"] > 1:
                student_obs = np.vstack([x["observation"][-1, :model.state_dims] for x in t_obs])
            else:
                student_obs = np.vstack([x["observation"][:model.state_dims] for x in t_obs])
            teacher_obs = t_obs[0]["teacher_frames"][t_obs[0]["teacher_attn_mask"]]
            rtg = np.cumsum(t_rew[::-1])[::-1] # numpy magic
            solved_trajectories["student"][t_info["traj_id"]] = {"observations": student_obs, "actions": t_act, "returns_to_go": rtg, "success": data["success"]}
            solved_trajectories["teacher"][t_info["traj_id"]] = {"observations": teacher_obs}
        return data



    ids_per_env = len(ids) // n_envs
    def make_env(idx):
        def _init():
            import tr2.envs
            env_kwargs = OmegaConf.to_container(env_cfg)
            env_kwargs["trajectories"] = ids[idx * ids_per_env: (idx + 1) * ids_per_env]
            if "planner_cfg" in env_kwargs and env_kwargs["planner_cfg"] is not None:
                if "BoxPusherTrajectory" in cfg.env:
                    from tr2.envs.boxpusher.env import BoxPusherEnv
                    from tr2.planner.boxpusherteacher import BoxPusherTaskPlanner
                    planner = BoxPusherTaskPlanner()
                    planning_env = BoxPusherEnv(
                        **env_kwargs["planner_cfg"]["env_cfg"]
                    )
                    planning_env.seed(0)
                    planner.set_type(1)
                    planning_env.reset()
                elif "BlockStack" in cfg.env:
                    from tr2.envs.blockstacking.env import BlockStackMagicPandaEnv
                    from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper
                    from tr2.planner.blockstackplanner import BlockStackPlanner
                    planner = BlockStackPlanner(replan_threshold=1e-1)
                    planning_env = BlockStackMagicPandaEnv(obs_mode="state_dict", goal=env_cfg.goal, num_blocks=1 if 'train' in env_cfg.goal else -1)
                    planning_env = ManiSkillActionWrapper(planning_env)
                    planning_env = NormalizeActionWrapper(planning_env)
                    planning_env.seed(0)
                    planning_env.reset()
                elif "OpenDrawer" in cfg.env:
                    from tr2.planner.opendrawerplanner import OpenDrawerPlanner
                    from mani_skill.env.open_cabinet_door_drawer import OpenCabinetDrawerMagicEnv_CabinetSelection
                    planner = OpenDrawerPlanner(replan_threshold=1e-2)
                    # we only need planning env to get details about the object used
                    planning_env = OpenCabinetDrawerMagicEnv_CabinetSelection()
                    planning_env.set_env_mode(obs_mode=env_cfg.obs_mode, reward_type='sparse')
                    planning_env.reset()
                env_kwargs["planner_cfg"]["planner"] = planner
                env_kwargs["planner_cfg"]["planning_env"] = planning_env
            env = gym.make(cfg.env, **env_kwargs)
            env.seed(10000*idx)
            return env
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(cfg.n_envs)])
    # env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: True if x % 50 == 0 else False, video_length=50)
    use_ac = False
    if use_ac: 
        model = TranslationTeacherStudentActorCritic(
            actor_model=model_cls.load_from_checkpoint(ckpt, device=device), 
            critic_model=model_cls.load_from_checkpoint(ckpt, device=device), action_space=env.action_space,
        )
        model.load_state_dict(ckpt["ac_state_dict"])
        print("### LOGSTD", model.pi.log_std)
        model=model.to(device)
        model.eval()
    def policy(o):
        with torch.no_grad():
            o = obs_to_tensor(o)
            if not use_teacher:
                o["teacher_attn_mask"][:] = False
                o["teacher_frames"] = o["teacher_frames"] * 0
            if not use_ac:
                a = model.step(o)
                a = model.actions_scaler.untransform(a).cpu().numpy()
            else:
                a = model.act(o, deterministic=False)
        return a
    render_mode = False
    if save_video:
        render_mode = "rgb_array"
    
    if "watch" in cfg and cfg.watch:
        render_mode = "human"
    video_imgs = []
    def video_capture(output, step):
        video_imgs.append(output)
    trajectories = rollout.collect_trajectories(
        policy=policy,
        env=env,
        n_trajectories=(len(ids) // n_envs) * n_envs,
        n_envs=n_envs,
        pbar=True,
        even_num_traj_per_env=True,
        format_trajectory=format_trajectory,
        render=render_mode,
        video_capture=video_capture if save_video else None,
    )
    if save_video:
        animate(video_imgs, "eval.mp4", fps=30)
        print("saved video")
    val_results = []
    for traj_set in trajectories:
        for traj in traj_set:
            val_results.append(traj)
    val_df = pd.DataFrame(val_results)
    def get_success_rates(df):
        successful_match_rate = (df['match_left'] < 5).sum() / len(df)
        successful_completion_rate = (df['match_left'] == 0).sum() / len(df)
        if not cfg.env_cfg.task_agnostic:
            successful_completion_rate = (df['success'] == True).sum() / len(df)
        return successful_match_rate, successful_completion_rate
    match_rate, complete_rate = get_success_rates(val_df)
    avg_traj_match = val_df["traj_match"].mean()
    add_info = ""
    print("FAILED", list(val_df[val_df['success'] == False]['traj_id']))
    print("SUCCESS", list(val_df[val_df['success'] == True]['traj_id']))
    if cfg.env_cfg.task_agnostic:
        add_info = "-task_agnostic"
    if cfg.ignore_teacher:
        add_info = "-noteacher"
    print(f"results: success_rate = {complete_rate}, avg return = {val_df['returns'].mean()}")
    dir_name = f"{osp.dirname(cfg.model)}"
    if dir_name == "":
        dir_name = "./"

    save_results_path = f"{dir_name}/{osp.basename(cfg.model)}-results{add_info}.csv"
    if "save_results_path" in cfg:
        save_results_path = cfg["save_results_path"]
        val_df.to_csv(save_results_path, index=False)
    env.close()
    if cfg.save_solved_trajectories:
        print(f"saving {len(solved_trajectories['student'])} solved trajectories")
        solved_trajectories_path = "eval_solved_dataset.pkl"
        with open(solved_trajectories_path, "wb") as f:
            pickle.dump(solved_trajectories, f)


    exit()


if __name__ == "__main__":
    base_cfgs_path = osp.join(osp.dirname(__file__), "../cfgs")
    cli_conf = OmegaConf.from_cli()
    custom_cfg = None
    if "cfg" in cli_conf: custom_cfg = cli_conf["cfg"]
    cfg = parse.parse_cfg(default_cfg_path=osp.join(base_cfgs_path, "defaults/eval_translation.yml"), cfg_path=custom_cfg)
    parse.clean_and_transform(cfg)
    main(cfg)