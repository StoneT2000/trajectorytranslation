import numpy as np
import torch
import torch.nn as nn


class TranslationPolicy(nn.Module):
    """
    Base translation model that takes in teacher frames and past + current student frames and produces actions.
    """
    def __init__(self, state_dims, act_dims, teacher_dims, max_length=None):
        super().__init__()

        self.state_dims = state_dims
        self.act_dims = act_dims
        self.teacher_dims = teacher_dims
        self.max_length = max_length

    def forward(self, teacher_frames, student_frames, teacher_attn_mask, student_attn_mask, teacher_time_steps, student_time_steps, returns_to_go, **kwargs):
        raise NotImplementedError()

    def step(self, obs, **kwargs):
        """
        expect dictionary input, two kinds

        obs = {
            "teacher_frames": (N_T, d)
            "teacher_attn_mask": (N_T),
            "teacher_time_steps": (N_T),
            "student_frames": (stack_size, d),
            "student_attn_mask": (stack_size),
            "student_time_steps": (N_T),
        }
        or
        obs = {
            "teacher_frames": (N_T, d)
            "teacher_attn_mask": (N_T),
            "teacher_time_steps": (N_T),
            "observation": (d),
            "step": (),
        }
        """
        raise NotImplementedError()
