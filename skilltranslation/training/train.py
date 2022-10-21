"""
Training class for training with BC. Not used in paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from skilltranslation.models.translation.model import TranslationPolicy


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        get_batch,
        loss_fnc,
        scheduler=None,
        train_cb=None,
        tensorboard=None, # if a string, logs to there
        device=torch.device("cpu")
    ) -> None:
        self.model = model
        self.optim = optimizer
        self.get_batch = get_batch
        self.loss_fnc = loss_fnc
        self.scheduler = scheduler
        self.train_cb = train_cb
        self.losses = []
        self.device = device
        self.steps = 0
        self.tb_writer = None
        if tensorboard is not None:
            self.tb_writer = SummaryWriter(log_dir=tensorboard)

    def train_step(self):
        x, y_gt, attn_mask, time_steps = self.get_batch()
        self.optim.zero_grad()

        y = self.model.forward(x.to(self.device), attention_mask=attn_mask.to(self.device), time_steps=time_steps.to(self.device))

        loss = self.loss_fnc(y, y_gt.to(self.device), attn_mask=attn_mask)

        loss.backward()
        self.optim.step()

        loss_itm = loss.detach().cpu().item()
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/loss", loss_itm, global_step=self.steps)
        if self.scheduler is not None:
            self.scheduler.step()
        self.losses.append(loss_itm)
        self.steps += 1

class TrainerTeacherStudent:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        get_batch,
        loss_fnc,
        state_loss_fnc=None,
        returns_to_go_loss_fnc=None,
        scheduler=None,
        train_cb=None,
        tensorboard=None, # if a string, logs to there
        device=torch.device("cpu"),
        log_freq=1,
        categorical=False,
        use_last_prediction_only=False, # when true, do less auto regressive like training
    ) -> None:
        self.use_last_prediction_only = use_last_prediction_only
        self.model: TranslationPolicy = model
        self.categorical = categorical
        self.optim = optimizer
        self.get_batch = get_batch
        self.loss_fnc = loss_fnc
        self.state_loss_fnc = state_loss_fnc
        self.returns_to_go_loss_fnc = returns_to_go_loss_fnc
        self.scheduler = scheduler
        self.train_cb = train_cb
        self.device = device
        self.steps = 0
        self.log_freq = log_freq
        self.tb_writer = None
        if tensorboard is not None:
            self.tb_writer = SummaryWriter(log_dir=tensorboard)
    
    def get_loss_from_batch(self, batch):
        teacher_traj, stacked_student_frames, teacher_attn_mask, student_attn_mask, teacher_time_steps, student_time_steps, target_student_actions, student_rtg = batch
        if student_rtg is not None:
            student_rtg = student_rtg.to(self.device)
        y = self.model.forward(
            teacher_traj.to(self.device),
            stacked_student_frames.to(self.device), 
            teacher_attn_mask.to(self.device), 
            student_attn_mask.to(self.device), 
            teacher_time_steps.to(self.device), 
            student_time_steps.to(self.device),
            returns_to_go=student_rtg,
            action_pred=True,
            state_pred=self.state_loss_fnc is not None,
            returns_to_go_pred=self.returns_to_go_loss_fnc is not None,

        ) # (B, stack_size, act_dim)
        
        y_action = y["action"]
        act_dim = y_action.shape[2]
        # if self.categorical:

        #     act_dim = target_student_actions.shape[1]
        # else:
        #     act_dim = target_student_actions.shape[2]
        # mask out padded tokens so we only look at the actual predictions.
        loss_dict = dict()
        with torch.no_grad():
            if self.categorical:
                loss_dict["last_action_loss"] = self.loss_fnc(y_action[:, -1], target_student_actions[:, -1].to(self.device))
                loss_dict["last_action_accuracy"] = (torch.sum(y_action[:, -1].argmax(1) == target_student_actions[:, -1].to(self.device)) / len(y_action))
            else:
                loss_dict["last_action_loss"] = self.loss_fnc(y_action[:, -1], target_student_actions[:, -1].to(self.device))
        if self.use_last_prediction_only:
            
            y_action = y_action[:, -1]
            # if self.categorical:
            #     y_action = F.softmax(y_action, dim=1)
            # print(y_action)
            target_student_actions = target_student_actions[:, -1]
            action_loss = self.loss_fnc(y_action, target_student_actions.to(self.device))
            loss_dict["last_action_loss"] = action_loss
        else:
            # print(student_attn_mask.shape, y_action.shape)
            y_action = y_action.reshape(-1, act_dim)[student_attn_mask.reshape(-1)]
            if self.categorical:
                target_student_actions = target_student_actions.reshape(-1)[student_attn_mask.reshape(-1)]
                loss_dict["action_accuracy"] = (torch.sum(y_action.argmax(1) == target_student_actions.to(self.device)) / len(y_action))
            else:
                target_student_actions = target_student_actions.reshape(-1, act_dim)[student_attn_mask.reshape(-1)]
            action_loss = self.loss_fnc(y_action, target_student_actions.to(self.device))
            
        loss_dict["action_loss"] = action_loss
       
        if self.state_loss_fnc is not None:
            raise NotImplementedError("State loss not done")
        if self.returns_to_go_loss_fnc is not None:
            raise NotImplementedError("Returns to go loss not done")
            # we can just make it so that the return should be the fraction of teacher trajectory achieved.
        loss = action_loss
        loss_dict["loss"] = loss
        return loss_dict

    def train_step(self):
        self.optim.zero_grad()
        loss_dict = self.get_loss_from_batch(self.get_batch())

        loss_dict["loss"].backward()
        self.optim.step()
        loss_dict_itm = dict()
        for k in loss_dict:
            loss_dict_itm[k] = loss_dict[k].cpu().item()
        if self.train_cb is not None: self.train_cb(step=self.steps, loss_dict=loss_dict_itm)
        if self.tb_writer is not None and self.steps % self.log_freq == 0:
            self.tb_writer.add_scalar("train/loss", loss_dict_itm["loss"], global_step=self.steps)
        if self.scheduler is not None:
            self.scheduler.step()
        self.steps += 1
