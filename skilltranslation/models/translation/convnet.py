"""
Convnet (1D filter) backbone
"""
from re import X
from torch.distributions.categorical import Categorical
import numpy as np
from skilltranslation.data.utils import MinMaxScaler
from skilltranslation.models.utils import act2fnc
import torch
import torch.nn as nn
import transformers
from gym.spaces import Box, Discrete
from paper_rl.architecture.ac.core import Actor, mlp
from torch.distributions.normal import Normal

from skilltranslation.models.translation.model import TranslationPolicy
from skilltranslation.models.translation.modelling_gpt2 import GPT2Model

class TranslationConvNet(TranslationPolicy):
    """
    Translates teacher sequence to a student sequence
    """

    @staticmethod
    def load_from_checkpoint(ckpt, device=torch.device("cpu")):
        """
        returns a instantiated model from a checkpoint

        checkpoint can be from either the online training or offline training
        """
        cfg = ckpt["cfg"]
        model_cfg = cfg["model_cfg"]
        model = TranslationConvNet(
            **model_cfg
        ).to(device)
        if "actions_scaler" in ckpt.keys():
            model.actions_scaler.min = torch.as_tensor(ckpt["actions_scaler"]["min"], dtype=torch.float32, device=device)
            model.actions_scaler.max = torch.as_tensor(ckpt["actions_scaler"]["max"], dtype=torch.float32, device=device)
        
        if "state_dict" in ckpt:
            # load bc trained model
            state_dict = ckpt["state_dict"]
            model.load_state_dict(
                state_dict
            )
        elif "ac_state_dict" in ckpt:
            # load online trained actor of actor critic model
            ac_weights = ckpt["ac_state_dict"]
            state_dict = model.state_dict()
            loaded = 0
            for k in ac_weights.keys():
                match = "pi.mu_net.model."
                if k[:len(match)] == match:
                    state_dict[k[len(match):]] = ac_weights[k]
                    loaded += 1
            assert len(state_dict) == loaded
            model.load_state_dict(
                state_dict
            )
        return model
    def __init__(
        self,
        state_dims,
        act_dims,
        teacher_dims,
        max_time_steps=1024,
        max_student_length=128,
        max_teacher_length=128,
        trajectory_sample_skip_steps=0,
        timestep_embeddings=True,
        use_past_actions=True,
        use_returns_to_go=False,
        embed_layer_norm=True,
        stack_size=5,
        state_embedding_hidden_sizes=(32,),
        state_embedding_activation='relu',
        final_mlp_hidden_sizes=(),
        final_mlp_activation='tanh',
        final_mlp_action_pred_activation='tanh',
        final_mlp_state_pred_activation='tanh',
        head_type='default',
        actions_scaler = {"min": -1, "max": 1},
        # GPT2 configs
        prepend_student=False,
        convnet_config=dict(
            n_layer=2,
            kernel_size=5,
            dropout=0.1,
        ),
        **kwargs
    ):
        """
        Parameters
        ----------
        state_dims, act_dims - correspond with env. state_dims does not include teacher trajectory, step information, purely a single observation

        embedding_activation - activation function (e.g. `nn.ReLU`) applied after embedding layers

        hidden_size - hidden size in GPT2 transformer
        """
        repeated_stack=1
        if prepend_student:
            repeated_stack += 1
        max_length = max_teacher_length + stack_size * repeated_stack
        if use_past_actions:
            max_length += stack_size * repeated_stack
        if use_returns_to_go:
            max_length += stack_size * repeated_stack
        for k in kwargs:
            print(f"TranslationConvNet Model given kwargs {k} not used")

        super().__init__(state_dims=state_dims, act_dims=act_dims, teacher_dims=teacher_dims, max_length=max_length)
        self.max_teacher_length = max_teacher_length
        self.prepend_student=prepend_student
        self.max_student_length = max_student_length
        self.trajectory_sample_skip_steps = trajectory_sample_skip_steps
        self.embed_layer_norm = embed_layer_norm
        self.use_past_actions = use_past_actions
        self.timestep_embeddings = timestep_embeddings
        self.head_type = head_type
        self.stack_size = stack_size
        self.state_embedding_size = state_embedding_hidden_sizes[-1]
        self.raw_convnet_config = convnet_config
        self.use_returns_to_go = use_returns_to_go
        self.actions_scaler = MinMaxScaler()
        self.actions_scaler.min = actions_scaler["min"]
        self.actions_scaler.max = actions_scaler["max"]


        if teacher_dims is not None:
            self.teacher_dims = teacher_dims
        else:
            print("teacher_dims arg not provided, defaulting setting to state_dims")
            self.teacher_dims = state_dims

        # 1d convnet to process a batch of sequences (B, seq_length, h) to (B, seq_length, h)
        self.convnet = nn.Conv1d(state_embedding_hidden_sizes[-1], state_embedding_hidden_sizes[-1], kernel_size=convnet_config['kernel_size'], stride=1, padding_mode="zeros", padding="same")
        self.dropout = nn.Dropout(p=convnet_config["dropout"])

        self.embed_teacher_timestep = nn.Embedding(max_time_steps, state_embedding_hidden_sizes[-1])
        self.embed_student_timestep = nn.Embedding(max_time_steps, state_embedding_hidden_sizes[-1])
        
        # state embedding for both teacher and student states for this work
        state_embedding_act = act2fnc(state_embedding_activation)
        self.embed_teacher_state = mlp([self.teacher_dims] + list(state_embedding_hidden_sizes), activation=state_embedding_act)
        
        self.embed_student_state = mlp([self.state_dims] + list(state_embedding_hidden_sizes), activation=state_embedding_act)

        if self.use_returns_to_go:
            self.embed_returns_to_go = mlp([1] + list(state_embedding_hidden_sizes), activation=state_embedding_act)

        # action embedding for student actions
        self.embed_student_action = mlp([self.act_dims] + list(state_embedding_hidden_sizes), activation=state_embedding_act) 

        self.embed_ln = nn.LayerNorm(state_embedding_hidden_sizes[-1])

        final_mlp_act = act2fnc(final_mlp_activation)
        final_mlp_action_pred_act = act2fnc(final_mlp_action_pred_activation)
        if final_mlp_action_pred_activation == "identity" or final_mlp_action_pred_activation == "":
            # identity action predictions, no need to scale.
            self.final_mlp_action_pred_activation = "identity"
        else:
            self.final_mlp_action_pred_activation = final_mlp_action_pred_activation
        final_mlp_state_pred_act = act2fnc(final_mlp_state_pred_activation)
        if head_type == "default":
            self.predict_action = mlp([state_embedding_hidden_sizes[-1]] + list(final_mlp_hidden_sizes) + [act_dims], activation=final_mlp_act, output_activation=final_mlp_action_pred_act)
            self.predict_state = mlp([state_embedding_hidden_sizes[-1]] + list(final_mlp_hidden_sizes) + [state_dims], activation=final_mlp_act, output_activation=final_mlp_state_pred_act)
            self.predict_returns_to_go = mlp([state_embedding_hidden_sizes[-1]] + list(final_mlp_hidden_sizes) + [1], activation=final_mlp_act, output_activation=nn.Identity)
        else:
            self.final_layer = head_type


    def forward(self, teacher_traj, stacked_student_frames, teacher_attn_mask, student_attn_mask, teacher_time_steps, student_time_steps, returns_to_go=None, output_attentions=False, action_pred=True, state_pred=False, returns_to_go_pred=False):
        """
        
        """
        batch_size, seq_length = teacher_traj.shape[0], teacher_traj.shape[1]
        stack_size = stacked_student_frames.shape[1]
            
        
        # ctx_act_embeddings = self.embed_action
        teacher_state_embeddings = self.embed_teacher_state(teacher_traj) # (B, seq_length, state_embedding_size)

        # stacked_student_frames is (B, stack_size, ...state...actions)

        student_state_embeddings = self.embed_student_state(stacked_student_frames[:,:,:self.state_dims]) # (B, stack_size, state_embedding_size)
        if self.use_past_actions:
            student_action_embeddings = self.embed_student_action(stacked_student_frames[:,:,self.state_dims:]) # (B, stack_size, state_embedding_size)
        if self.timestep_embeddings:
            teacher_pos_embeddings = self.embed_teacher_timestep(teacher_time_steps) # (B, seq_length, state_embedding_size)
            student_pos_embeddings = self.embed_student_timestep(student_time_steps) # (B, stack_size, state_embedding_size)
            teacher_state_embeddings = teacher_state_embeddings + teacher_pos_embeddings
            student_state_embeddings = student_state_embeddings + student_pos_embeddings
            if self.use_past_actions:
                student_action_embeddings = student_action_embeddings + student_pos_embeddings
        if self.use_returns_to_go:
            if len(returns_to_go.shape) == 2:
                returns_to_go = returns_to_go[:,:,None]
            returns_to_go_embeddings = self.embed_returns_to_go(returns_to_go) # (B, stack_size, state_embedding_size)
        """
        embedding: 
        [teacher_state_embeds...., (s_t, a_t, s_t+1, a_t,...)] 
        """
        embeddings_arr = []
        if self.use_returns_to_go: embeddings_arr.append(returns_to_go_embeddings)
        embeddings_arr.append(student_state_embeddings)
        if self.use_past_actions: embeddings_arr.append(student_action_embeddings)
        modalities = len(embeddings_arr)
        if modalities > 1:
            stacked_student_attn_mask = torch.stack([student_attn_mask for _ in range(modalities)], 1).permute(0, 2, 1).reshape(batch_size, modalities * self.stack_size)
        else:
            stacked_student_attn_mask = student_attn_mask
        
        # (B, stack_size * 2, hidden_state) - (s_t, a_t, s_t+1, a_t+1,...)
        if modalities > 1:
            stacked_student_inputs = torch.stack(embeddings_arr, dim=2).reshape(batch_size, self.stack_size * modalities, self.state_embedding_size)
        else:
            stacked_student_inputs = student_state_embeddings # (B, stack_size, hidden_state)
       
        if self.prepend_student:
            attention_mask = torch.hstack([stacked_student_attn_mask, teacher_attn_mask, stacked_student_attn_mask])
            stacked_inputs = torch.hstack(
                [stacked_student_inputs, teacher_state_embeddings, stacked_student_inputs]
            )
        else:
            attention_mask = torch.hstack([teacher_attn_mask, stacked_student_attn_mask])
            stacked_inputs = torch.hstack(
                [teacher_state_embeddings, stacked_student_inputs]
        )

        if self.embed_layer_norm:
            stacked_inputs = self.embed_ln(stacked_inputs)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        stacked_inputs = stacked_inputs.permute(0, 2, 1) # convnet requires first dim = batch, 2nd dim = hidden size, 3rd dim is sequence
        x = self.convnet(stacked_inputs)
        x = self.dropout(x)
        x = x.permute(0, 2, 1) # revert back to the (B, seq_length, hidden_size) order
        # context vector x
        student_x = x[:, self.max_length - self.stack_size * modalities:, ]
        student_x = student_x.reshape(batch_size, stack_size, modalities, self.state_embedding_size).permute(0, 2, 1, 3)
        # student_x[:, 0] is hidden states after each state, student_x[:, 1] is hidden state after each action
        # if returns to go tokens are used, student_x[:, 1] is hidden states after each state
        state_hidden_states = student_x[:, 0]
        if self.use_past_actions:
            state_action_hidden_states = student_x[:, 1]
        if self.use_returns_to_go:
            state_hidden_states = student_x[:, 1]
            if self.use_past_actions:
                state_action_hidden_states = student_x[:, 2]

        if self.head_type == "default":
            to_return = dict()
            if action_pred:
                # predict actions given the transformer outputs of the teacher states, compared with student ground truth actions
                action_preds = self.predict_action(state_hidden_states) # (B, stack_size, act_dim)
                to_return["action"] = action_preds
            if output_attentions:
                to_return["transformer_outputs"] = None
            if state_pred:
                # predict the next state given the past actions and states
                state_preds = self.predict_state(state_action_hidden_states) # (B, stack_size, state_dim)
                to_return["state"] = state_preds
            if returns_to_go_pred:
                # predict the returns to go given state
                returns_to_go_preds = self.predict_returns_to_go(state_hidden_states)
                to_return["returns_to_go"] = returns_to_go_preds
            return to_return
        else:
            y = self.final_layer(student_x[:, 0])
            if output_attentions:
                return y, None
            return y

    def format_obs(self, obs):
        device = next(self.parameters()).device
        prestacked = False
        if "observation_attn_mask" in obs.keys():
            prestacked = True
        batch_size = len(obs["observation"])

        teacher_traj = torch.as_tensor(obs["teacher_frames"], dtype=torch.float32, device=device)
        if prestacked:
            stacked_student_frames = obs["observation"]
        else:
            stacked_student_frames = torch.zeros((batch_size, self.stack_size, self.state_dims + self.act_dims), device=device)
            stacked_student_frames[:, :, self.state_dims:] = -10.
            stacked_student_frames[:, -1, :self.state_dims] = torch.as_tensor(obs["observation"], device=device)

        teacher_attn_mask = torch.as_tensor(obs["teacher_attn_mask"], dtype=torch.bool, device=device)

        if prestacked:
            student_attn_mask = obs["observation_attn_mask"]
        else:
            student_attn_mask = torch.zeros((batch_size, self.stack_size), dtype=torch.bool, device=device)
            student_attn_mask[:, -1] = 1
        
        teacher_time_steps = torch.as_tensor(obs["teacher_time_steps"], dtype=torch.long, device=device)
        if prestacked:
            student_time_steps = obs["observation_time_steps"]
        else:
            student_time_steps = torch.zeros((batch_size, self.stack_size), dtype=torch.long, device=device)
            student_time_steps[:, -1] = torch.as_tensor(obs["step"], dtype=torch.long, device=device)

        student_rtg = None
        if self.use_returns_to_go:
            # if prestacked:
                # should always 0 to maximize performance due to LCS_DP with step=-1 implementation
            student_rtg = torch.zeros((batch_size, self.stack_size), dtype=torch.float32)
        return teacher_traj, stacked_student_frames, teacher_attn_mask, student_attn_mask, teacher_time_steps, student_time_steps, student_rtg
    
    def step(self, obs, output_attentions=False):
        device = next(self.parameters()).device
        teacher_traj, stacked_student_frames, teacher_attn_mask, student_attn_mask, teacher_time_steps, student_time_steps, student_rtg = self.format_obs(obs)
        if self.use_returns_to_go:
            if student_rtg is None:
                student_rtg = torch.zeros_like(student_time_steps, dtype=torch.float32).to(device) - 1
            else:
                student_rtg = student_rtg.to(device)
        if output_attentions:
            res = self.forward(
                teacher_traj.to(device),
                stacked_student_frames.to(device), 
                teacher_attn_mask.to(device), 
                student_attn_mask.to(device),
                teacher_time_steps.to(device), 
                student_time_steps.to(device),
                returns_to_go=student_rtg,
                output_attentions=True,
                action_pred=True,
            )
            return res["action"][:, -1], res["transformer_outputs"]
        else:
            res = self.forward(
                teacher_traj.to(device),
                stacked_student_frames.to(device), 
                teacher_attn_mask.to(device),
                student_attn_mask.to(device),
                teacher_time_steps.to(device), 
                student_time_steps.to(device),
                returns_to_go=student_rtg,
                action_pred=True,
                output_attentions=False,
            )
            return res["action"][:, -1]
