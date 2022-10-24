"""
MLP backbone
"""

import numpy as np
import torch
import torch.nn as nn
import transformers
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from paper_rl.architecture.ac.core import Actor, mlp
from skilltranslation.data.utils import MinMaxScaler
from skilltranslation.models.translation.model import TranslationPolicy
from skilltranslation.models.utils import act2fnc


class MLPTranslationID(TranslationPolicy):
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
        model = MLPTranslationID(
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
                match2 = "pi.logits_net."
                if k[:len(match)] == match:
                    state_dict[k[len(match):]] = ac_weights[k]
                    loaded += 1
                elif k[:len(match2)] == match2:
                    state_dict[k[len(match2):]] = ac_weights[k]
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
        mlp_config=dict(
            hidden_sizes=(32, 32),
            max_embedding=2000,
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
        for k in kwargs:
            print(f"MLP Model given kwargs {k} not used")
        max_length = 1
        assert stack_size == 1
        super().__init__(state_dims=state_dims, act_dims=act_dims, teacher_dims=teacher_dims, max_length=max_length)
        # following not used, kept just for consistency with other models
        self.max_teacher_length = max_teacher_length
        self.max_student_length = max_student_length
        self.trajectory_sample_skip_steps = trajectory_sample_skip_steps
        self.embed_layer_norm = embed_layer_norm
        self.use_past_actions = use_past_actions
        self.timestep_embeddings = timestep_embeddings
        self.head_type = head_type
        self.stack_size = stack_size
        self.state_embedding_size = state_embedding_hidden_sizes[-1]
        self.raw_mlp_config = mlp_config
        self.actions_scaler = MinMaxScaler()
        self.actions_scaler.min = actions_scaler["min"]
        self.actions_scaler.max = actions_scaler["max"]

        state_embedding_act = act2fnc(state_embedding_activation)
        self.embed_student_state = mlp([self.state_dims] + list(state_embedding_hidden_sizes), activation=state_embedding_act, output_activation=state_embedding_act)
        # self.id_embedding = nn.Embedding(mlp_config["max_embedding"], embedding_dim=mlp_config["embedding_dim"])
        self.dropout = None
        if mlp_config["dropout"] > 0:
            self.dropout = nn.Dropout(mlp_config["dropout"])

        self.embed_timestep = nn.Embedding(max_time_steps, state_embedding_hidden_sizes[-1])

        # # action embedding for student actions
        # self.embed_student_action = mlp([self.act_dims] + list(state_embedding_hidden_sizes), activation=state_embedding_activation) 

        final_mlp_act = act2fnc(final_mlp_activation)
        final_mlp_action_pred_act = act2fnc(final_mlp_action_pred_activation)
        if final_mlp_action_pred_activation == "identity" or final_mlp_action_pred_activation == "":
            # identity action predictions, no need to scale.
            self.final_mlp_action_pred_activation = "identity"
        else:
            self.final_mlp_action_pred_activation = final_mlp_action_pred_activation
        final_mlp_state_pred_act = act2fnc(final_mlp_state_pred_activation)
        if head_type == "default":
            self.predict_action = mlp([state_embedding_hidden_sizes[-1] ] + list(final_mlp_hidden_sizes) + [act_dims], activation=final_mlp_act, output_activation=final_mlp_action_pred_act)
            self.predict_state = mlp([state_embedding_hidden_sizes[-1] ] + list(final_mlp_hidden_sizes) + [state_dims], activation=final_mlp_act, output_activation=final_mlp_state_pred_act)
            self.predict_returns_to_go = mlp([state_embedding_hidden_sizes[-1] ] + list(final_mlp_hidden_sizes) + [1], activation=final_mlp_act, output_activation=nn.Identity)
        else:
            self.final_layer = head_type

    def forward(self, obs, traj_id, student_time_steps=None, output_attentions=False, action_pred=True, state_pred=False, returns_to_go_pred=False):
        """
        
        """
        batch_size = obs.shape[0]
        # stack_size = stacked_student_frames.shape[1]
        # stack size should be size 1
        # id_context = self.id_embedding(traj_id)
        embedding = self.embed_student_state(obs)
        if self.timestep_embeddings:
            embedding = embedding + self.embed_timestep(student_time_steps)[:, 0]
        # if len(id_context.shape) == 3:
            # id_context = id_context.reshape(batch_size, self.raw_mlp_config["embedding_dim"])
        # x = torch.hstack([embedding, id_context]) 
        x = embedding
        if self.head_type == "default":
            to_return = dict()
            if action_pred:
                # predict actions given the transformer outputs of the teacher states, compared with student ground truth actions
                
                action_preds = self.predict_action(x) # (B, stack_size, act_dim)
                to_return["action"] = action_preds
            if output_attentions:
                to_return["transformer_outputs"] = None
            if state_pred:
                # predict the next state given the past actions and states
                state_preds = self.predict_state(x) # (B, stack_size, state_dim)
                to_return["state"] = state_preds
            if returns_to_go_pred:
                # predict the returns to go given state
                returns_to_go_preds = self.predict_returns_to_go(x)
                to_return["returns_to_go"] = returns_to_go_preds
            return to_return
        else:
            y = self.final_layer(x)
            if output_attentions:
                return y, None
            return y

    def format_obs(self, obs):
        device = next(self.parameters()).device
        prestacked = False
        batch_size = len(obs["observation"])
        if "observation_attn_mask" in obs.keys():
            prestacked = True
        if "traj_id" in obs:
            traj_id = obs["traj_id"]
        else:
            traj_id = None
        if prestacked:
            student_time_steps = obs["observation_time_steps"]
        else:
            student_time_steps = torch.zeros((batch_size, self.stack_size), dtype=torch.long, device=device)
            student_time_steps[:] = torch.as_tensor(obs["step"].reshape(batch_size, self.stack_size), dtype=torch.long, device=device)
        return torch.as_tensor(obs["observation"], dtype=torch.float32, device=device), student_time_steps, traj_id
        
    def step(self, obs, output_attentions=False):
        device = next(self.parameters()).device
        obs, student_time_steps, traj_id = self.format_obs(obs)
        if output_attentions:
            res = self.forward(
                obs, traj_id,
                student_time_steps=student_time_steps,
                output_attentions=True,
                action_pred=True,
            )
            return res["action"], res["transformer_outputs"]
        else:
            res = self.forward(
                obs, traj_id,
                student_time_steps=student_time_steps,
                action_pred=True,
                output_attentions=False,
            )
            return res["action"]
class MLPTranslationIDTeacherStudentCategoricalActor(Actor):
    def __init__(self, model: MLPTranslationID):
        super().__init__()
        # self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.logits_net: MLPTranslationID = model

    def _distribution(self, obs):
        logits = self.logits_net.step(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        act = act.reshape(-1)
        return pi.log_prob(act)

    def act(self, obs):
        
        logits = self.logits_net.step(obs)
        return logits.argmax(1)
class MLPTranslationIDTeacherStudentGaussianActorMuNet(nn.Module):
    def __init__(self, model: MLPTranslationID, actions_scaler=None) -> None:
        super().__init__()
        self.model = model
        self.actions_scaler = actions_scaler
    def forward(self, obs):
        a = self.model.step(obs)
        if self.actions_scaler is not None:
            a = self.actions_scaler.untransform(a,)
        return a
        
class MLPTranslationIDTeacherStudentGaussianActor(Actor):
    def __init__(self, model, act_dim, actions_scaler=None, log_std_scale=-0.5):
        super().__init__()
        
        log_std = log_std_scale * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = MLPTranslationIDTeacherStudentGaussianActorMuNet(model, actions_scaler)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        device = next(self.mu_net.parameters()).device
        return pi.log_prob(act.to(device)).sum(
            axis=-1
        )
    def act(self, obs):
        mu = self.mu_net(obs)
        return mu


class MLPTranslationIDTeacherStudentCritic(nn.Module):
    def __init__(self, critic_model):
        super().__init__()
        self.v_net: MLPTranslationID = critic_model

    def forward(self, obs):
        device = next(self.v_net.parameters()).device
        obs, student_time_steps, traj_id = self.v_net.format_obs(obs)
        res = self.v_net.forward(
            obs, traj_id,
            student_time_steps=student_time_steps,
            action_pred=False,
            returns_to_go_pred=True,
            output_attentions=False,
        )
        return torch.squeeze(
            res["returns_to_go"][:, -1], -1
        )


class MLPTranslationIDTeacherStudentActorCritic(nn.Module):
    def __init__(
        self,
        actor_model,
        critic_model,
        action_space,
        actions_scaler=None,
        log_std_scale=-0.5,
    ):
        super().__init__()
        self.actions_scaler = actions_scaler
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPTranslationIDTeacherStudentGaussianActor(
                actor_model, action_space.shape[0], actions_scaler, log_std_scale
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPTranslationIDTeacherStudentCategoricalActor(
                model=actor_model
            )

        # build value function
        self.v = MLPTranslationIDTeacherStudentCritic(critic_model)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs, deterministic=False):
        if deterministic:
            return self.pi.act(obs).cpu().numpy()
        return self.step(obs)[0]