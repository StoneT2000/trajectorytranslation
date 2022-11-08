"""
Training a TR2-GPT2 model with behavior cloning. Not used in the main paper but works
"""

from collections import defaultdict
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from omegaconf import DictConfig, OmegaConf
from paper_rl.cfg.parse import clean_and_transform, parse_cfg
from paper_rl.logger.logger import Logger
from tqdm import tqdm
from tr2.envs.evaluate import evaluate_online
from tr2.models.translation.convnet import TranslationConvNet

import tr2.training.train as train
from tr2.data.teacherstudent import TeacherStudentDataset
from tr2.models.translation.lstm import LSTM
from tr2.models.translation.model import TranslationPolicy
from tr2.models.translation.translation_transformer import \
    TranslationTransformerGPT2


def main(cfg):
    exp_cfg = cfg["exp_cfg"]
    model_cfg = cfg["model_cfg"]
    logging_cfg = cfg["logging_cfg"]
    dataset_cfg = cfg["dataset_cfg"]

    exp_name = logging_cfg.exp_name
    exp_dir = osp.join(logging_cfg.workspace, exp_name)
    log_dir = osp.join(exp_dir, "logs")
    configs_file = osp.join(exp_dir, "config.yml")
    model_save_path = osp.join(exp_dir, "models")
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

    logger = Logger(
        wandb=logging_cfg.wandb,
        tensorboard=logging_cfg.tensorboard,
        workspace=logging_cfg.workspace,
        exp_name=logging_cfg.exp_name,
        project_name="tr2",
        cfg=cfg,
        clear_out=new_session
    )
    Path(model_save_path).mkdir(parents=True, exist_ok=True)


    seed = exp_cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("====train_translation_bc configuration===\n", OmegaConf.to_yaml(cfg))
    print(f"====load dataset====")

    dataset_cfg["stack_size"] = model_cfg.stack_size
    dataset_cfg["trajectory_sample_skip_steps"] = model_cfg.trajectory_sample_skip_steps
    dataset_cfg["max_teacher_length"] = model_cfg.max_teacher_length
    dataset_cfg["max_student_length"] = model_cfg.max_student_length

    train_dataset, val_dataset = TeacherStudentDataset.create_train_val_sets(**dataset_cfg)
    batch_size = exp_cfg["batch_size"]
    train_dl = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, collate_fn=train_dataset.collate_fn, drop_last=exp_cfg.drop_last
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size, collate_fn=val_dataset.collate_fn
    )
    is_categorical = train_dataset.is_categorical
    if not is_categorical:
        actions_scaler_info = {"max": train_dataset.actions_scaler.max, "min": train_dataset.actions_scaler.min}
    else:
        actions_scaler_info = {"max": 1, "min": -1}
    print(f"====loaded dataset====")
    def get_batch():
        return next(iter(train_dl))

    device = torch.device(cfg["device"])
    model: TranslationPolicy = None
    print(f"====building model {model_cfg['type']}====")
    if cfg.pretrained_weights is not None:
        print(f"====building model {model_cfg['type']} with pretrained weights {cfg.pretrained_weights}====")
        ckpt = torch.load(cfg.pretrained_weights, map_location=device)
        model_cfg = OmegaConf.create(ckpt["cfg"]["model_cfg"])
        if model_cfg["type"] == "TranslationTransformer":
            model = TranslationTransformerGPT2(
                actions_scaler=actions_scaler_info,
                **model_cfg)
        elif model_cfg["type"] == "LSTM":
            model = LSTM(
                actions_scaler=actions_scaler_info,
                **model_cfg)
        model.load_state_dict(ckpt["state_dict"])
    else:
        if model_cfg["type"] == "TranslationTransformer":
            model = TranslationTransformerGPT2(
                actions_scaler=actions_scaler_info,
                **model_cfg)
        elif model_cfg["type"] == "LSTM":
            model = LSTM(
                actions_scaler=actions_scaler_info,
                **model_cfg)
        elif model_cfg["type"] == "ConvNet":
            model = TranslationConvNet(
                actions_scaler=actions_scaler_info,
                **model_cfg)
    model = model.to(device)
    print(f"====built model {model_cfg['type']}====")

    start_step = 0
    optim = torch.optim.Adam(model.parameters(), lr=exp_cfg["lr"])
    scheduler = None
    best_eval_loss_dict = defaultdict(lambda: None)
    if not new_session:
        if osp.exists(osp.join(model_save_path, "latest.pt")):
            
            ckpt = torch.load(osp.join(model_save_path, "latest.pt"))
            model.load_state_dict(ckpt["state_dict"])
            optim.load_state_dict(ckpt["optim_state_dict"])
            start_step = ckpt["step"]
            print("found previous model from step " + str(start_step))
            best_eval_loss_dict = ckpt["best_eval_loss_dict"]
            if scheduler is not None:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    def loss_fnc(y, y_gt):
        diffs = (y - y_gt) ** 2
        return torch.mean(diffs)
    if train_dataset.is_categorical:
        loss_fnc = torch.nn.CrossEntropyLoss()
    
    def train_cb(step, loss_dict):
        if step % logging_cfg.log_freq == 0: logger.store(tag="train", append=False, **loss_dict)

    trainer = train.TrainerTeacherStudent(
        model=model,
        optimizer=optim,
        get_batch=get_batch,
        loss_fnc=loss_fnc,
        tensorboard=None,
        train_cb=train_cb,
        device=device,
        log_freq=logging_cfg.log_freq,
        use_last_prediction_only=exp_cfg.predict_current_state_only,
        scheduler=scheduler,
        categorical=is_categorical,
    )
    def evaluate(model, val_dl):
        if cfg.eval_cfg is not None:
            base_cfgs_path = osp.join(osp.dirname(__file__), "../cfgs")
            eval_cfg = OmegaConf.load(osp.join(base_cfgs_path, "defaults/eval_translation.yml"))
            user_eval_cfg = OmegaConf.load(cfg.eval_cfg)
            eval_cfg.merge_with(user_eval_cfg)
            ids = list(np.load(eval_cfg.env_cfg.trajectories))[:]
            id_rng = np.random.default_rng(0)
            id_rng.shuffle(ids)
            ids=ids[:eval_cfg.test_n]
            model.eval()
            eval_cfg.env_cfg["stack_size"] = model_cfg.stack_size
            eval_cfg.env_cfg["max_trajectory_length"] = model_cfg.max_teacher_length
            eval_cfg.env_cfg["trajectory_sample_skip_steps"] = model_cfg.trajectory_sample_skip_steps
            def eval_policy(o):
                with torch.no_grad():
                    a = model.step(o)
                    a = model.actions_scaler.untransform(a.cpu().numpy())
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
        else:            
            losses = defaultdict(list)
            loss_dict_avg = dict()
            with torch.no_grad():
                i = 0
                for batch in val_dl:
                    loss_dict = trainer.get_loss_from_batch(batch)
                    for k, v in loss_dict.items():
                        losses[k].append(v.detach().cpu().item())

            for k, v in losses.items():
                loss_dict_avg[k] = np.mean(v)
            eval_results = loss_dict_avg
        model.train()
        return eval_results

    assert exp_cfg.eval_freq % logging_cfg.log_freq == 0
    pbar = tqdm(range(start_step, exp_cfg.steps))
    for i in pbar:
        if "save_freq" in exp_cfg and exp_cfg.save_freq != -1 and i % exp_cfg.save_freq == 0 and i != start_step:
            save_data = {
                "step": i,
                "state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "cfg": OmegaConf.to_container(cfg),
                "actions_scaler": actions_scaler_info,
                "best_eval_loss_dict": dict(best_eval_loss_dict)
            }
            torch.save(save_data, osp.join(model_save_path, f"model_{i}.pt"))
        if i % exp_cfg.eval_freq == 0 and i != start_step:
            save_data = {
                "step": i,
                "state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "cfg": OmegaConf.to_container(cfg),
                "actions_scaler": actions_scaler_info
            }
            if scheduler is not None:
                save_data["scheduler_state_dict"] = scheduler.state_dict(),

            # evaluate model and save
            eval_loss_dict = evaluate(model, val_dl)
            logger.store(tag="test", append=False, **eval_loss_dict)
            save_data["eval_loss_dict"] = eval_loss_dict
            improved_on_keys = set()
            for k in eval_loss_dict:
                if k not in best_eval_loss_dict:
                    best_eval_loss_dict[k] = eval_loss_dict[k]
                    improved_on_keys.add(k)
                else:
                    improved = eval_loss_dict[k] < best_eval_loss_dict[k]
                    if "accuracy" in k or "success_rate" in k or "avg_traj_match" in k or "avg_return" in k:
                        improved=eval_loss_dict[k] > best_eval_loss_dict[k]
                    if improved:
                        best_eval_loss_dict[k] = eval_loss_dict[k]
                        improved_on_keys.add(k)
            save_data["best_eval_loss_dict"] = dict(best_eval_loss_dict)
            for k in improved_on_keys:
                print(f"Improved {k}, saving")
                torch.save(save_data, osp.join(model_save_path, f"best_{k}.pt"))
            torch.save(save_data, osp.join(model_save_path, "latest.pt"))

        trainer.train_step()
        if i % logging_cfg.log_freq == 0: 
            stats = logger.log(step=i)
            logger.reset()
            pbar.set_postfix({**stats})


if __name__ == "__main__":
    base_cfgs_path = osp.join(osp.dirname(__file__), "../cfgs")
    cli_conf = OmegaConf.from_cli()
    custom_cfg = None
    if "cfg" in cli_conf: custom_cfg = cli_conf["cfg"]
    cfg = parse_cfg(default_cfg_path=osp.join(base_cfgs_path, "defaults/train_translation_bc.yml"), cfg_path=custom_cfg)
    # convenience for common parameters when using CLI
    for k in ["dataset", "train_ids", "val_ids"]:
        if k in cfg:
            cfg.dataset_cfg[k] = cfg[k]
    for k in ["state_dims", "act_dims", "teacher_dims"]:
        if k in cfg:
            cfg.model_cfg[k] = cfg[k]
    clean_and_transform(cfg)
    main(cfg)
