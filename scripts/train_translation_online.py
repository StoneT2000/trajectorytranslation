from pathlib import Path
import pickle
import pandas as pd
from skilltranslation.envs.evaluate import evaluate_online
from skilltranslation.models.translation.lstm import LSTM, LSTMTeacherStudentActorCritic
from skilltranslation.models.translation.mlp_id import MLPTranslationID, MLPTranslationIDTeacherStudentActorCritic
from skilltranslation.utils.animate import animate
import sys
import gym
import numpy as np
from skilltranslation.data.teacherstudent import TeacherStudentDataset
from skilltranslation.models.translation.translation_transformer import TranslationTeacherStudentActorCritic, TranslationTransformerGPT2
from skilltranslation.utils.tools import merge
import torch
import torch.utils.data
from omegaconf import DictConfig, OmegaConf
from paper_rl.cfg.parse import clean_and_transform, parse_cfg
from paper_rl.logger.logger import Logger
from paper_rl.modelfree.ppo import PPO
import os.path as osp
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
def main(cfg):
    env_cfg = cfg["env_cfg"]
    exp_cfg = cfg["exp_cfg"]
    model_cfg = cfg["model_cfg"]
    logging_cfg = cfg["logging_cfg"]

    exp_name = logging_cfg.exp_name
    exp_dir = osp.join(logging_cfg.workspace, exp_name)
    configs_file = osp.join(exp_dir, "config.yml")
    # if True, wipes out old logs and weights, starts from scratch
    #  if False, does above if there aren't existing logs and weights, otherwise tries to continue training
    restart_training = False
    if "restart_training" in cfg:
        restart_training  = cfg["restart_training"]
    new_session = True
    if not restart_training:
        # look for previous config
        print("===attempting to resume training...===")
        if osp.exists(configs_file):
            print(f"===previous config found, resuming training for {logging_cfg.workspace}/{exp_name}===")
            cfg = OmegaConf.load(configs_file)
            new_session = False
        else:
            print(f"===previous config not found, starting training for {logging_cfg.workspace}/{exp_name}===")


    # select algorithm
    if exp_cfg.algo == "ppo":
        if exp_cfg.dapg:
            print("=== Running PPO / DAPG ===")
            assert exp_cfg.dapg_cfg.trajectories_dataset is not None
        else:
            print("=== Running PPO ===")

    device = torch.device(cfg.device)
    torch.manual_seed(exp_cfg.seed)
    np.random.seed(exp_cfg.seed)

    # build models, load from ckpts if necessary
    model_cls = TranslationTransformerGPT2
    if model_cfg["type"] == "LSTM":
        model_cls = LSTM
    elif model_cfg["type"] == "MLPID":
        model_cls = MLPTranslationID
    
    print("===loaded actor model===")
    given_actor_model_cfg = None
    if new_session and model_cfg.pretrained_actor_weights is not None:
        print("using pretrained actor weights " + model_cfg.pretrained_actor_weights)
        # only use pretrained weights provided if this is a new training session. Otherwise rely on previous checkpoint if resuming training
        ckpt = torch.load(model_cfg.pretrained_actor_weights, map_location=device)
        given_actor_model_cfg = ckpt["cfg"]["model_cfg"]
        model_cls = TranslationTransformerGPT2
        if given_actor_model_cfg["type"] == "LSTM":
            model_cls = LSTM
        elif given_actor_model_cfg["type"] == "MLPID":
            model_cls = MLPTranslationID
        actor_model = model_cls(
            **ckpt["cfg"]["model_cfg"]
        )
        
        actor_model.load_state_dict(ckpt["state_dict"])
        actor_model.actions_scaler.min = torch.as_tensor(ckpt["actions_scaler"]["min"], dtype=torch.float32, device=device)
        actor_model.actions_scaler.max = torch.as_tensor(ckpt["actions_scaler"]["max"], dtype=torch.float32, device=device)
    else:
        actor_model = model_cls(
            **OmegaConf.to_container(model_cfg)
        )
    
    if new_session and model_cfg.pretrained_critic_weights is not None:
        print("using pretrained critic weights " + model_cfg.pretrained_critic_weights)
        # usually not provided, or is the same path as actor weights if pretrained weights were also trained with returns to go losses
        ckpt = torch.load(model_cfg.pretrained_critic_weights, map_location=device)
        given_actor_model_cfg = ckpt["cfg"]["model_cfg"]
        model_cls = TranslationTransformerGPT2
        if given_actor_model_cfg["type"] == "LSTM":
            model_cls = LSTM
        elif given_actor_model_cfg["type"] == "MLPID":
            model_cls = MLPTranslationID
        critic_model = model_cls(
            **ckpt["cfg"]["model_cfg"]
        )
        critic_model.load_state_dict(ckpt["state_dict"])
        critic_model.actions_scaler.min = torch.as_tensor(ckpt["actions_scaler"]["min"], dtype=torch.float32, device=device)
        critic_model.actions_scaler.max = torch.as_tensor(ckpt["actions_scaler"]["max"], dtype=torch.float32, device=device)
    else:
        if given_actor_model_cfg is not None:
            print("No critic pretrained weights given, using config for given actor model weights")
            critic_model = model_cls(
                **OmegaConf.to_container(given_actor_model_cfg)
            )
        else:
            print("No critic pretrained weights given, using random init")
            critic_model = model_cls(
                **OmegaConf.to_container(model_cfg)
            )
    if given_actor_model_cfg is not None:
        model_cfg = merge(dict(model_cfg), given_actor_model_cfg)
        cfg["model_cfg"] = model_cfg
    print("===loaded critic model===")

    if isinstance(env_cfg.trajectories, str):
        trajectories = list(np.load(env_cfg.trajectories))
    else:
        trajectories = list(env_cfg.trajectories)
    env_cfg.stack_size = model_cfg["stack_size"]
    env_cfg.max_trajectory_length = model_cfg["max_teacher_length"]
    env_cfg.trajectory_sample_skip_steps = model_cfg["trajectory_sample_skip_steps"]
    env_cfg.fixed_max_ep_len = exp_cfg.max_ep_len
    env_cfg.offscreen_only = True

    # auto filled in
    # env_cfg["teacher_dims"] = model_cfg["teacher_dims"]
    # env_cfg["state_dims"] = model_cfg["state_dims"]
    # env_cfg["act_dims"] = model_cfg["act_dims"]
    import skilltranslation.envs
    def make_env(idx):
        def _init():
            import skilltranslation.envs
            del env_cfg.trajectories
            env = gym.make(
                cfg.env, 
                trajectories=trajectories,
                **env_cfg
            )
            env.seed(idx)
            return env

        return _init
    if exp_cfg.n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(exp_cfg.n_envs)])
    else:
        env = DummyVecEnv([make_env(i) for i in range(exp_cfg.n_envs)])
    print("===setup parallel envs===")


    if exp_cfg.dapg:
        assert exp_cfg.dapg_cfg.train_ids is not None, "Missing trajectory ids to use for dapg"
        assert exp_cfg.dapg_cfg.trajectories_dataset is not None, "Missing trajectories dataset to use for dapg"
        train_demos_dataset, _ = TeacherStudentDataset.create_train_val_sets(
            dataset=exp_cfg.dapg_cfg.trajectories_dataset,
            max_student_length=model_cfg["max_student_length"],
            max_teacher_length=model_cfg["max_teacher_length"],
            stack_size=model_cfg["stack_size"],
            scaled_actions=False,
            train_ids=exp_cfg.dapg_cfg.train_ids,
            val_ids=exp_cfg.dapg_cfg.train_ids,
            trajectory_sample_skip_steps=model_cfg["trajectory_sample_skip_steps"]
        )
        train_demo_dl = torch.utils.data.DataLoader(
            train_demos_dataset, shuffle=True, batch_size=exp_cfg.batch_size, collate_fn=train_demos_dataset.collate_fn
        )
        def demo_trajectory_sampler(batch_size):
            teacher_traj, stacked_student_frames, teacher_attn_mask, student_attn_mask, teacher_time_steps, student_time_steps, target_student_actions, student_rtg = next(iter(train_demo_dl))
            if train_demos_dataset.is_categorical:
                tgt = target_student_actions[:, -1]
            else:
                tgt = target_student_actions[:, -1, :]
            return dict(
                observations=dict(
                    teacher_frames=teacher_traj,
                    teacher_time_steps=teacher_time_steps,
                    teacher_attn_mask=teacher_attn_mask,
                    observation=stacked_student_frames,
                    observation_attn_mask=student_attn_mask,
                    observation_time_steps=student_time_steps,
                ),
                actions=tgt# slice out the actual action, shape (B, act_dims)
            )
        print("===setup dapg data sampler===")
    
    if model_cfg["type"] == "TranslationTransformer":
        ac = TranslationTeacherStudentActorCritic(
            actor_model=actor_model,
            critic_model=critic_model,
            action_space=env.action_space,
            actions_scaler=actor_model.actions_scaler,
            log_std_scale=exp_cfg.log_std_scale,
        ).to(device)
    elif model_cfg["type"] == "MLPID":
        ac = MLPTranslationIDTeacherStudentActorCritic(
            actor_model=actor_model,
            critic_model=critic_model,
            action_space=env.action_space,
            actions_scaler=actor_model.actions_scaler,
            log_std_scale=exp_cfg.log_std_scale,
        ).to(device)
    elif model_cfg["type"] == "LSTM":
        ac = LSTMTeacherStudentActorCritic(
            actor_model=actor_model,
            critic_model=critic_model,
            action_space=env.action_space,
            actions_scaler=actor_model.actions_scaler,
            log_std_scale=exp_cfg.log_std_scale,
        ).to(device)
    if "pretrained_ac_weights" in cfg:
        weight_path = cfg["pretrained_ac_weights"]
        print(f"===loading an AC model from: {weight_path}===")
        state_dict = torch.load(weight_path)["ac_state_dict"]
        # state_dict["pi.log_std"] = state_dict["pi.log_std"] * 0.7
        ac.load_state_dict(state_dict)
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=exp_cfg.pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=exp_cfg.vf_lr)
    raw_data = []
    wandb_cfg = None
    if "wandb_cfg" in logging_cfg:
        wandb_cfg = logging_cfg.wandb_cfg
    logger = Logger(
        tensorboard=logging_cfg.tensorboard,
        wandb=logging_cfg.wandb,
        workspace=logging_cfg.workspace,
        exp_name=logging_cfg.exp_name,
        project_name="skilltranslation",
        cfg=cfg,
        clear_out=new_session,
        wandb_cfg=wandb_cfg
    )
    model_save_path = osp.join(logger.exp_path, "models")
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    actions_scaler_info = {"min": actor_model.actions_scaler.min, "max": actor_model.actions_scaler.max}
    start_epoch = 0
    best_train_ret_avg = -100000
    best_eval_ret_avg = -100000
    best_eval_success_rate = -1
    if not new_session:
        ckpt_path = osp.join(model_save_path, "latest.pt")
        if osp.exists(ckpt_path):
            print("===loading AC checkpoint...===")
            ac_ckpt = torch.load(ckpt_path, map_location=device)
            ac.load_state_dict(ac_ckpt["ac_state_dict"])
            pi_optimizer.load_state_dict(ac_ckpt["pi_optimizer_state_dict"])
            vf_optimizer.load_state_dict(ac_ckpt["vf_optimizer_state_dict"])
            actions_scaler_info = ac_ckpt["actions_scaler"]
            start_epoch = ac_ckpt["epoch"] + 1
            best_train_ret_avg = ac_ckpt["best_train_ret_avg"]

            print("===loaded AC checkpoint===")
        else:
            print("===can't resume training as no model checkpoint found, exiting. Please re-run with restart_training=True===")
            exit()


    def obs_to_tensor(o):
        tensor_o = {}
        for k in o:
            tensor_o[k] = torch.as_tensor(o[k], device=device)
            if tensor_o[k].dtype == torch.float64:
                tensor_o[k] = tensor_o[k].float()
        return tensor_o

    def unsqueeze_dict(o):
        for k in o:
            o[k] = o[k].unsqueeze(0)
        return o

    def rollout_callback(observations, next_observations, actions, rewards, infos, dones, timeouts):
        for idx, info in enumerate(infos):
            if timeouts[idx] or dones[idx]:
                if timeouts[idx]:
                    logger.store("train", append=True, Success=0)
                else:
                    logger.store("train", append=True, Success=1)
                if "stats" in info.keys():
                    data = {}
                    for k, v in info["stats"].items():
                        data[k] = v
                    logger.store("train", append=True, **data)

    latest_stats = {}
    if hasattr(env_cfg, 'trajectories'):
        del env_cfg.trajectories
    eval_env = gym.make(cfg.env, trajectories=trajectories, **env_cfg)
    eval_env.seed(exp_cfg.seed + 10000)
    
    def evaluate(epoch, deterministic=True, save_video=False):
        if cfg.eval_cfg is not None:
            base_cfgs_path = osp.join(osp.dirname(__file__), "../cfgs")
            eval_cfg = OmegaConf.load(osp.join(base_cfgs_path, "defaults/eval_translation.yml"))
            user_eval_cfg = OmegaConf.load(cfg.eval_cfg)
            eval_cfg.merge_with(user_eval_cfg)
            ids = list(np.load(eval_cfg.env_cfg.trajectories))[:]
            id_rng = np.random.default_rng(0)
            id_rng.shuffle(ids)
            ids=ids[:eval_cfg.test_n]
            ac.eval()
            eval_cfg.env_cfg["stack_size"] = model_cfg["stack_size"]
            eval_cfg.env_cfg["max_trajectory_length"] = model_cfg["max_teacher_length"]
            eval_cfg.env_cfg["trajectory_sample_skip_steps"] = model_cfg["trajectory_sample_skip_steps"]
            def eval_policy(o):
                with torch.no_grad():
                    a = ac.act(o, deterministic=deterministic)
                return a
            eval_results = evaluate_online(
                env_name=eval_cfg.env,
                n_envs=eval_cfg.n_envs,
                env_cfg=eval_cfg.env_cfg,
                ids=ids,
                policy=eval_policy,
                device=device,
                use_teacher=True,
                noise=eval_cfg.noise,
                render=False
            )
            
            del eval_results["df"]
            print(eval_results)
            logger.store("test", append=False, **eval_results)
            ac.train()
            return None
        else:
            obs = eval_env.reset()
            done = False
            imgs = []
            ep_len = 0
            observations = [obs]
            rewards = []
            returns = 0
            actions = []
            ac.eval()
            info = dict()

            while not done and ep_len < env_cfg.fixed_max_ep_len:
                obs = obs_to_tensor(obs)

                with torch.no_grad():
                    obs = unsqueeze_dict(obs)
                    action = ac.act(obs, deterministic=deterministic)
                    if action.shape[0] == 1:
                        action = action[0]

                obs, reward, done, info = eval_env.step(action)
                observations.append(obs)
                rewards.append(reward)
                returns += reward
                actions.append(action)
                ep_len += 1
                if save_video:
                    if "OpenDrawer" in cfg.env:
                        img = eval_env.render(mode="color_image")["world"]["rgb"]
                        img = img * 255
                        img = img.astype(int)
                    else:
                        img = eval_env.render(mode="rgb_array")


                    imgs.append(img)
                if done:
                    logger.store("test", append=False, EpLen=ep_len, EpRet=returns)
                    if "stats" in info.keys():
                        data = {}
                        for k, v in info["stats"].items():
                            data[k] = v
                        logger.store("test", append=False, **data)
            if save_video:
                animate(imgs, filename=osp.join(logger.exp_path, f"eval_{epoch}.mp4"), _return=False, fps=24)
            logger.store("test", append=False, EpRet=np.sum(rewards), EpLen=ep_len)
            ac.train()
            return {
                "observations": np.vstack(observations),
                "actions": np.vstack(actions),
                "rewards": np.vstack(rewards),
                "returns_to_go": np.cumsum(rewards[::-1])[::-1],
                "last_info": info,
                "success": info["task_complete"]
            }
        
        return eval_trajectory

    def train_callback(epoch):
        nonlocal latest_stats, best_train_ret_avg, best_eval_success_rate, best_eval_ret_avg
        eval_trajectory = None
        if epoch % exp_cfg.eval_freq == 0:
            eval_trajectory = evaluate(
                epoch, deterministic=True, save_video=exp_cfg.eval_save_video
            )
        log_std_mean = ac.pi.log_std.detach().cpu().numpy().mean()
        logger.store("model", append=False, log_std_mean=log_std_mean)
        latest_stats = logger.log(step=epoch, local_only = epoch % logging_cfg.log_freq != 0)
        filtered = {}
        for k in latest_stats.keys():
            if (
                "Epoch" in k
                or "TotalEnvInteractions" in k
                or "EpRet" in k
                or "EpLen" in k
                # or "VVals" in k
                or "LossDAPGActor_avg" in k
                # or "PPOLoss_avg" in k
                # or "dapg_lambda" in k
                or "LossPi_avg" in k
                # or "KL_avg" in k
                # or "ClipFrac_avg" in k
                or "UpdateTime" in k
                or "RolloutTime" in k
                # or "farthest_traj_match_frac" in k
                # or "closest_traj_match_frac" in k
                or "lcs" in k
                or "Success" in k
            ):
                filtered[k] = latest_stats[k]
        if cfg.verbose == 1:
            logger.pretty_print_table(filtered)
        to_csv_keys = [
            "train/EpLen_avg",
            "train/EpRet_avg",
            "train/farthest_traj_match_frac_avg",
            "train/farthest_traj_match_frac_max",
            "train/RolloutTime",
            "train/UpdateTime",
            "train/Success_avg",
        ]
        csv_data = {}
        for k in to_csv_keys:
            csv_data[k] = latest_stats[k]
        raw_data.append(csv_data)
        logger.reset()
        if eval_trajectory is not None:
            with open(osp.join(logger.exp_path, "latest_traj.pkl"), "wb") as f:
                pickle.dump(eval_trajectory, f)

        if exp_cfg.save_model:  
            save_data = {
                "ac_state_dict": ac.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "stats": latest_stats.copy(),
                "actions_scaler": actions_scaler_info,
                "pi_optimizer_state_dict": pi_optimizer.state_dict(),
                "vf_optimizer_state_dict": vf_optimizer.state_dict(),
                "best_train_ret_avg": best_train_ret_avg
            }
            if best_train_ret_avg < filtered["train/EpRet_avg"]:
                best_train_ret_avg = filtered["train/EpRet_avg"]
                save_data["best_train_ret_avg"] = best_train_ret_avg
                torch.save(save_data, osp.join(logger.exp_path, f"models/best_train_EpRet.pt"))
            if "test/success_rate" in latest_stats:
                if best_eval_success_rate > latest_stats["test/success_rate"]:
                    best_eval_success_rate = latest_stats["test/success_rate"]
                    torch.save(save_data, osp.join(logger.exp_path, f"models/best_test_success_rate.pt"))
            if "test/avg_return" in latest_stats:
                if best_eval_ret_avg < latest_stats["test/avg_return"]:
                    best_eval_ret_avg = latest_stats["test/avg_return"]
                    torch.save(save_data, osp.join(logger.exp_path, f"models/best_test_success_rate.pt"))
                
        if exp_cfg.save_model and exp_cfg.save_freq > 0 and epoch % exp_cfg.save_freq == 0:
            save_data = {
                "ac_state_dict": ac.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "stats": latest_stats.copy(),
                "actions_scaler": actions_scaler_info,
                "pi_optimizer_state_dict": pi_optimizer.state_dict(),
                "vf_optimizer_state_dict": vf_optimizer.state_dict(),
                "best_train_ret_avg": best_train_ret_avg
            }
            torch.save(save_data, osp.join(logger.exp_path, f"models/latest.pt"))

        if exp_cfg.good_test_trajectory_threshold is not None and eval_trajectory is not None and eval_trajectory["success"]:
            teacher_length = eval_trajectory['last_info']['traj_len']
            student_length = len(eval_trajectory['observations'])
            if teacher_length * exp_cfg.good_test_trajectory_threshold > student_length:
                solved_traj_path = osp.join(logger.exp_path, "solved_traj.pkl")
                with open(solved_traj_path, "wb") as f:
                    pickle.dump(eval_trajectory, f)
                print(f"Got trajectory of length {student_length} < teacher_length*{exp_cfg.good_test_trajectory_threshold} = {teacher_length}, stored at {solved_traj_path}, finishing up now")
                eval_env.close()
                # exit()
                return True

        return False

    steps_per_epoch = exp_cfg.steps_per_epoch // exp_cfg.n_envs
    batch_size = exp_cfg.batch_size
    algo = PPO(
        ac=ac,
        env=env,
        num_envs=exp_cfg.n_envs,
        action_space=env.action_space,
        observation_space=env.observation_space,
        logger=logger,
        steps_per_epoch=steps_per_epoch,
        train_iters=exp_cfg.update_iters,
        gamma=exp_cfg.gamma,
        device=device,
        target_kl=exp_cfg["target_kl"],
        pi_coef=exp_cfg["pi_coef"],
        dapg_lambda=exp_cfg.dapg_cfg["dapg_lambda"],
        dapg_damping=exp_cfg.dapg_cfg["dapg_damping"],
    )
    # evaluate(0, True, True)
    if not exp_cfg.dapg:
        demo_trajectory_sampler = None
    algo.train(
        pi_optimizer=pi_optimizer,
        vf_optimizer=vf_optimizer,
        max_ep_len=exp_cfg.max_ep_len,
        start_epoch=start_epoch,
        n_epochs=exp_cfg.epochs - start_epoch,
        batch_size=batch_size,
        rollout_callback=rollout_callback,
        train_callback=train_callback,
        accumulate_grads=exp_cfg.accumulate_grads,
        obs_to_tensor=obs_to_tensor,
        critic_warmup_epochs=exp_cfg.critic_warmup_epochs,
        dapg_nll_loss=exp_cfg.dapg_cfg["dapg_nll_loss"],
        demo_trajectory_sampler=demo_trajectory_sampler,
        
    )
    df = pd.DataFrame(raw_data)
    df.to_csv(osp.join(logger.exp_path, "metrics.csv"))
    env.close()
    eval_env.close()
    import sys
    sys.exit()
    
    

if __name__ == "__main__":
    base_cfgs_path = osp.join(osp.dirname(__file__), "../cfgs")
    cli_conf = OmegaConf.from_cli()
    custom_cfg = None
    if "cfg" in cli_conf: custom_cfg = cli_conf["cfg"]

    cfg = parse_cfg(default_cfg_path=osp.join(base_cfgs_path, "defaults/train_translation_online.yml"), cfg_path=custom_cfg)

    # convenience for common parameters when using CLI
    for k in ["state_dims", "act_dims", "teacher_dims"]:
        if k in cfg:
            cfg.model_cfg[k] = cfg[k]
    clean_and_transform(cfg)
    main(cfg)
