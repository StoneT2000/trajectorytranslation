import pickle
import time
import numpy as np
from omegaconf import OmegaConf
from skilltranslation.data.teacherstudent import TeacherStudentDataset
from skilltranslation.data.utils import MinMaxScaler
from skilltranslation.models.translation.lstm import LSTM
import gym
from paper_rl.cfg import parse
from skilltranslation.models.translation.mlp_id import MLPTranslationID, MLPTranslationIDTeacherStudentActorCritic
import torch
from skilltranslation.models.translation.translation_transformer import (
    TranslationTeacherStudentActorCritic,
    TranslationTransformerGPT2,
)
from skilltranslation.utils.animate import animate
import os.path as osp
import skilltranslation.envs.boxpusher.traj_env
# import skilltranslation.envs.maniskill.traj_env
import skilltranslation.envs.maze.traj_env

def main(cfg):

    env_cfg = cfg.env_cfg
    model, traj_actions = None, None
    save_dir = cfg.save_dir
    save_traj_path = None
    if "save_traj_path" in cfg:
        save_traj_path = cfg["save_traj_path"]

    if cfg.model is not None:
        ckpt = torch.load(cfg.model, map_location=cfg.device)

        model_cfg = ckpt["cfg"]["model_cfg"]

        env_cfg.stack_size = model_cfg["stack_size"]
        env_cfg.max_trajectory_length = model_cfg["max_teacher_length"]
        env_cfg.trajectory_sample_skip_steps = model_cfg["trajectory_sample_skip_steps"]
        device = torch.device(cfg.device)
        model_cls = TranslationTransformerGPT2
        if model_cfg["type"] == "LSTM":
            model_cls = LSTM
        elif model_cfg["type"] == "MLPID":
            model_cls = MLPTranslationID
        model = model_cls.load_from_checkpoint(ckpt, device=device)
        model.eval()
        actions_scaler = None
        # print("###MODEL STEP: ", ckpt["stats"]["train/Epoch"])
        if "actions_scaler" in ckpt.keys():
            actions_scaler = MinMaxScaler()
            actions_scaler.min = torch.as_tensor(ckpt["actions_scaler"]["min"], dtype=torch.float32, device=device)
            actions_scaler.max = torch.as_tensor(ckpt["actions_scaler"]["max"], dtype=torch.float32, device=device)

        
    else:
        assert "traj" in cfg # require a working trajectory
        with open(cfg.traj, "rb") as f:
            data = pickle.load(f)
            if "student" in data.keys():
                traj_actions = data['student'][str(cfg.traj_id)]["actions"]
            else:
                traj_actions = data['actions']
        env_cfg.stack_size = 1
        env_cfg.max_trajectory_length = 1000 #exp_cfg.max_ep_len
    env_cfg.trajectories = [cfg.traj_id]

    save_video=cfg.save_video
    use_teacher=not cfg.ignore_teacher

    done = False
    
    env = gym.make(cfg.env, **env_cfg)
    use_ac = True
    if use_ac: 
        model = TranslationTeacherStudentActorCritic(
            actor_model=model_cls.load_from_checkpoint(ckpt, device=device), 
            critic_model=model_cls.load_from_checkpoint(ckpt, device=device), action_space=env.action_space, actions_scaler=actions_scaler,
        )
        # ckpt["ac_state_dict"]["pi.log_std"] = ckpt["ac_state_dict"]["pi.log_std"] * 0.5
        model.load_state_dict(ckpt["ac_state_dict"])
        print("### LOGSTD", model.pi.log_std)
        model=model.to(device)
        model.eval()
    env.seed(0)
    obs = env.reset()
    # if "trajectory_sample_skip_steps" in env_cfg:
    # env.draw_teacher_trajectory(skip=0)

    def obs_to_tensor(o):
        for k in o:
            o[k] = torch.as_tensor(o[k], device=device)
            if o[k].dtype == torch.float64:
                o[k] = o[k].float()
        return o

   
    ep_len=0
    attns = []
    trajectory = dict(
        states=[obs["observation"]],
        observations=[obs],
        actions=[]
    )
    # for layer in range(model_cfg["transformer_config"]["n_layer"]):
    #     attns.append([])
    imgs = []

    noise_generator = np.random.default_rng(0)
    ep_ret = 0
    success = False
    if not cfg.save_video:
        viewer = env.render("human")
        if "BlockStack" in cfg.env or "OpenDrawer" in cfg.env:
            viewer.paused=True
        if hasattr(env, "viewer"):
            env.viewer.paused=True
    success_step = -1
    while ep_len < 10000:
        # env.draw_teacher_trajectory(skip=0)
        if "traj_id" in obs:
            obs["traj_id"] = 0
        for k in obs:
            if not isinstance(obs[k], int) and not isinstance(obs[k], float):
                obs[k] = obs[k][None, :]
            else:
                new_k = torch.zeros((1, 1))
                new_k[0,0] = obs[k]
                obs[k] = new_k
        if not use_teacher:
            obs["teacher_attn_mask"][:] = False
            obs["teacher_frames"] = obs["teacher_frames"] * 0
        with torch.no_grad():
            if model is not None:
                obs=obs_to_tensor(obs)
                
                if not use_ac:
                    if cfg.save_attn:
                        a, attn = model.step(obs, output_attentions=True)
                        attns.append(attn)
                        layer=3
                        attn_vec = attn.attentions[layer][0].mean(0)
                        attn_vec = attn_vec[-1, -model.max_teacher_length-model.stack_size:-model.stack_size].cpu().numpy()
                        attn_vec[:]
                        env.visualize_attn(attn_vec)
                    else:
                        a = model.step(obs)
                    # a[0,0] = 0
                    if actions_scaler is not None:
                        a = actions_scaler.untransform(a)[0].cpu().numpy()
                    else:
                        a = a.cpu().numpy()[0]
                    # print("ACTION", a)
                else:
                    a = model.act(obs, deterministic=False)[0]
                    # print("AC", a)
                # if "Maze" in cfg.env:
                #     a[:2] = 0.175*a[:2] / np.linalg.norm(a[:2])
                #     a = a + noise_generator.normal(0, 2e-1, size=(3,))
            if traj_actions is not None:
                if ep_len >= len(traj_actions):
                    a = np.zeros_like(traj_actions[0])
                else:
                    a = traj_actions[min(ep_len, len(traj_actions)-1)]
        if save_video:
            if "ManiSkill" in cfg.env or "OpenDrawer" in cfg.env:
                img = env.render(mode="color_image")["world"]["rgb"]
                img = img * 255
            elif "BlockStack" in cfg.env:
                img = env.render(mode="state_visual")
                img = img["rgb"]
            else:
                img = env.render(mode="rgb_array")
                # img = img["rgb"]
            imgs.append(img)
        else:
            env.render("human")
            time.sleep(0.025)
            # if ep_len > 20: input()
        obs, reward, done, info = env.step(a)
        ep_ret += reward

        # trajectory["observations"].append(obs)
        trajectory["states"].append(obs["observation"])
        trajectory["actions"].append(a)

        print("Step:",ep_len, "Done:",info["task_complete"], reward, info['stats'])
        ep_len += 1
        if ep_len >= env_cfg.fixed_max_ep_len:
            break
        if info["task_complete"]:
            if success_step == -1:
                success_step = ep_len
                success = True
            if ep_len - success_step > 25:
                break
    if cfg.save_attn:
        attns = np.stack(attns)
        with open(f"attn_{cfg.traj_id}.pkl", "wb") as f:
            pickle.dump(attns, f)
    print(f"episode return: {ep_ret}, success: {success}")
    if save_video:
        
        print("animate", imgs[0].shape)
        save_name = osp.join(save_dir, f"execute_{cfg.traj_id}.mp4")
        print(f"saving video to {save_name}")
        animate(imgs, filename=save_name, _return=False, fps=20)
    env.close()

    if save_traj_path is not None:
        print(f"saving trajectory to {save_traj_path}")
        with open(save_traj_path, "wb") as f:
            pickle.dump(trajectory, f)

if __name__ == "__main__":
    base_cfgs_path = osp.join(osp.dirname(__file__), "../cfgs")
    cli_conf = OmegaConf.from_cli()
    custom_cfg = None
    if "cfg" in cli_conf: custom_cfg = cli_conf["cfg"]

    cfg = parse.parse_cfg(default_cfg_path=osp.join(base_cfgs_path, "defaults/watch_translation.yml"), cfg_path=custom_cfg)
    parse.clean_and_transform(cfg)
    main(cfg)