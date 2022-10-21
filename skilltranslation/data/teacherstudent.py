import numpy as np
import torch

from skilltranslation.data.utils import MinMaxScaler, get_trajectory_pairs


# from torch.utils.data import Dataset
class TeacherStudentDataset(torch.utils.data.Dataset):
    """
    dataset that gives teacher trajectory, student trajectory, teacher mask, and student mask

    Teacher trajectories are of shape (N_T, state_dim)
    Student trajectories are of shape (T_S, state_dim + act_dim)

    stack_size is number of context student frames to provide in the data, >= 1 (with 1 meaning only using the current frame)

    objective with this dataset is to use the full teacher trajectory, a sequence of past student obs actions + the current obs,
    and produce the next student (obs?) or action


    v2: items are 

    teacher_traj, stacked_student_frames, teacher_attn_masks student_attn_mask, teacher_time_steps, student_time_steps, target_student_actions

    using everything to predict target_student_actions
    """
    @staticmethod
    def create_train_val_sets(
        dataset=None,
        max_teacher_length=150,
        stack_size=1,
        scaled_actions=True,
        train_ids=None,
        val_ids=None,
        split_ratio=0.8,
        seed=0,
        trajectory_sample_skip_steps=0,
        max_student_length=None, # if none, notused. Otherwise sets a max length for student trajectories
    ):
        if train_ids is not None and val_ids is not None:
            def load_ids(ids):
                if isinstance(ids, str):
                    return list(np.load(ids))
                return ids
            train_ids = load_ids(train_ids)
            val_ids = load_ids(val_ids)
            train_trajectory_pairs, train_trajectory_ids, val_trajectory_pairs, val_trajectory_ids = get_trajectory_pairs(
                path=dataset, max_length=max_teacher_length, verbose=True, train_ids=train_ids,val_ids=val_ids,
                trajectory_sample_skip_steps=trajectory_sample_skip_steps
            )
        else:
            train_trajectory_pairs, train_trajectory_ids, val_trajectory_pairs, val_trajectory_ids = get_trajectory_pairs(
                path=dataset, max_length=max_teacher_length, verbose=True, train_test_split=True, split_seed=seed, split_ratio=split_ratio,
                trajectory_sample_skip_steps=trajectory_sample_skip_steps
            )
        state_dims = train_trajectory_pairs[0]["student_obs"].shape[1]
        is_categorical = False
        print(train_trajectory_pairs[0]["student_acts"].dtype)
        if np.issubdtype(train_trajectory_pairs[0]["student_acts"].dtype, np.integer):
            is_categorical = True
            print("Categorical actions detected, loading as categorical dataset, no action scaling performed")
            act_dims = 1
        else:
            act_dims = train_trajectory_pairs[0]["student_acts"].shape[1]
        teacher_dims = train_trajectory_pairs[0]["teacher"].shape[1]
        # create scaler here for both datasets.
        y_mins = np.zeros(act_dims)
        y_maxes = np.zeros(act_dims)
        for trajectory_pairs in [train_trajectory_pairs, val_trajectory_pairs]:
            for pair in trajectory_pairs:
                if is_categorical:
                    y_mins = np.min(
                        np.hstack([pair["student_acts"], y_mins]), 0
                    )
                    y_maxes = np.max(
                        np.hstack([pair["student_acts"], y_maxes]), 0
                    )
                else:    
                    y_mins = np.min(
                        np.vstack([pair["student_acts"], y_mins]), 0
                    )
                    y_maxes = np.max(
                    np.vstack([pair["student_acts"], y_maxes]), 0
                    )
        print("detected categorical range", y_mins, y_maxes)
        if is_categorical:
            scaled_actions = False
        actions_scaler = MinMaxScaler()
        for i in range(len(y_mins)):
            if y_mins[i] == 0 and y_maxes[i] == 0:
                y_mins[i] = -1
                y_maxes[i] = 1
                print(f"Action dim {i} is always zero")
        actions_scaler.min = y_mins
        actions_scaler.max = y_maxes


        train_dataset = TeacherStudentDataset(
            trajectory_pairs=train_trajectory_pairs,
            trajectory_ids=train_trajectory_ids,
            max_teacher_length=max_teacher_length,
            stack_size=stack_size,
            scaled_actions=scaled_actions,
            actions_scaler=actions_scaler,
            max_student_length=max_student_length,
        )
        if len(val_trajectory_ids) == 0:
            print("Empty val ids, loading only train data")
            return train_dataset, None
        val_dataset = TeacherStudentDataset(
            trajectory_pairs=val_trajectory_pairs,
            trajectory_ids=val_trajectory_ids,
            max_teacher_length=max_teacher_length,
            stack_size=stack_size,
            scaled_actions=scaled_actions,
            actions_scaler=actions_scaler,
            max_student_length=max_student_length,
        )
        
        return train_dataset, val_dataset


    def __init__(
        self,
        trajectory_pairs=None,
        trajectory_ids=None,
        max_teacher_length=150,
        stack_size=1,
        scaled_actions=True,  # scales actions to range [-1, 1]
        actions_scaler=None,
        max_student_length=None,
    ) -> None:
        super().__init__()
        self.stack_size = stack_size
        self.max_teacher_length = max_teacher_length
        self.max_student_length = max_teacher_length
        if max_student_length is not None:
            self.max_student_length = max_student_length
        # trajectory_ids is a list of string ids (which are all ints actually)
        # element i of trajectory pairs is [S_T, S_GT], trajectory of teacher (T) and trajectory of student (GT)
        self.trajectory_pairs, self.trajectory_ids = trajectory_pairs, trajectory_ids

        self.is_categorical = False
        if np.issubdtype(trajectory_pairs[0]["student_acts"].dtype, np.integer):
            self.is_categorical = True
            self.act_dim = int(actions_scaler.max) + 1
            print(self.act_dim + 1, "categories")
        else:
            self.act_dim = trajectory_pairs[0]["student_acts"].shape[1]

        self.state_dim = self.trajectory_pairs[0]["student_obs"].shape[1]
        # self.act_dim = self.trajectory_pairs[0]["student_acts"].shape[1]
        self.teacher_dim = self.trajectory_pairs[0]["teacher"].shape[1]
        print("Created dataset", f"state_dim={self.state_dim}, act_dim={self.act_dim}, teacher_dim={self.teacher_dim}")

        self.student_trajectory_lengths = torch.zeros((len(trajectory_pairs)), dtype=torch.long)
        self.teacher_trajectory_lengths = torch.zeros((len(trajectory_pairs)), dtype=torch.long)

        # create padded versions

        # (N, seq_len, state_dim)
        self.teacher_trajectories = np.zeros((len(trajectory_pairs), max_teacher_length, self.teacher_dim))
        # (N, seq_len, state_dim + act_dim) - [o_t, a_t], action a_t taken upon seeing o_t
        self.student_trajectories = np.zeros((len(trajectory_pairs), self.max_student_length, self.state_dim + self.act_dim))
        # (N, seq_len)
        self.teacher_attn_masks = np.zeros((len(trajectory_pairs), max_teacher_length), dtype=int)
        # (N, seq_len)
        self.student_attn_masks = np.zeros((len(trajectory_pairs), self.max_student_length), dtype=int)
        #
        self.use_returns_to_go = "student_rtg" in self.trajectory_pairs[0]
        print(f"Using returns to go: {self.use_returns_to_go}")
        if self.use_returns_to_go:
            self.student_rtgs = np.zeros((len(trajectory_pairs), self.max_student_length), dtype=np.float32)

        for i, (d) in enumerate(self.trajectory_pairs):
            teacher = d["teacher"]
            student_obs = d["student_obs"]
            student_acts = d["student_acts"]
            if self.use_returns_to_go:
                student_rtg = d['student_rtg']
                self.student_rtgs[i, self.max_student_length - len(student_rtg):] = student_rtg
            # convert actions to one hot
            if self.is_categorical:
                b = np.zeros((student_acts.size, self.act_dim))
                b[np.arange(student_acts.size), student_acts] = 1
                student_acts = b
            self.teacher_trajectories[i, max_teacher_length - len(teacher):] = teacher
            self.teacher_trajectory_lengths[i] = len(teacher)
            self.student_trajectories[i,  self.max_student_length - len(student_obs):] = np.hstack([student_obs, student_acts])
            self.student_trajectory_lengths[i] = len(student_obs)
            self.teacher_attn_masks[i, max_teacher_length - len(teacher):] = 1
            self.student_attn_masks[i, self.max_student_length - len(student_obs):] = 1

        # scale actions of student trajectory to [-1, 1]
        if scaled_actions:
            assert self.is_categorical == False
            if actions_scaler is not None:
                self.actions_scaler = actions_scaler
            else:
                y_mins = self.student_trajectories[self.student_attn_masks.astype(bool), self.state_dim:].min(0)
                y_maxes = self.student_trajectories[self.student_attn_masks.astype(bool), self.state_dim:].max(0)
                self.actions_scaler = MinMaxScaler()
                self.actions_scaler.min = y_mins
                self.actions_scaler.max = y_maxes
            student_act_scaled = self.actions_scaler.transform(self.student_trajectories[:,:,self.state_dim:])
            student_act_scaled[~self.student_attn_masks.astype(bool)] = 0
            self.student_trajectories[:,:,self.state_dim:] = student_act_scaled
            
            # set masked out actions to -10, -10....
            self.student_trajectories[:,:,self.state_dim:][~self.student_attn_masks.astype(bool)] = -10.

        self.teacher_trajectories = torch.from_numpy(self.teacher_trajectories).float()
        self.student_trajectories = torch.from_numpy(self.student_trajectories).float()
        self.teacher_attn_masks = torch.from_numpy(self.teacher_attn_masks).bool()
        self.student_attn_masks = torch.from_numpy(self.student_attn_masks).bool()
        if self.use_returns_to_go:
            self.student_rtgs = torch.from_numpy(self.student_rtgs).float()
    def __len__(self):
        return len(self.trajectory_ids) * self.max_student_length


    def collate_fn(self, batch):
        batch_size = len(batch)
        teacher_traj_b = torch.zeros((batch_size, self.max_teacher_length, self.teacher_dim))
        stacked_student_frames_b = torch.zeros((batch_size, self.stack_size, self.state_dim + self.act_dim))
        teacher_attn_mask_b = torch.zeros((batch_size, self.max_teacher_length), dtype=torch.bool)
        student_attn_mask_b = torch.zeros((batch_size, self.stack_size), dtype=torch.bool)
        teacher_time_steps_b = torch.zeros((batch_size, self.max_teacher_length), dtype=torch.long)
        student_time_steps_b = torch.zeros((batch_size, self.stack_size), dtype=torch.long)
        student_rtg_b = None
        if self.use_returns_to_go:
            student_rtg_b = torch.zeros((batch_size, self.stack_size), dtype=torch.float32)
        if self.is_categorical:
            target_student_actions_b = torch.zeros((batch_size, self.stack_size), dtype=torch.long)
        else:
            target_student_actions_b = torch.zeros((batch_size, self.stack_size, self.act_dim))

        for idx, (teacher_traj, stacked_student_frames, teacher_attn_mask, student_attn_mask, teacher_time_steps, student_time_steps, target_student_actions, student_rtg) in enumerate(batch):
            teacher_traj_b[idx] = teacher_traj
            stacked_student_frames_b[idx] = stacked_student_frames
            teacher_attn_mask_b[idx] = teacher_attn_mask
            student_attn_mask_b[idx] = student_attn_mask
            teacher_time_steps_b[idx] = teacher_time_steps
            student_time_steps_b[idx] = student_time_steps
            target_student_actions_b[idx] = target_student_actions
            if self.use_returns_to_go:
                student_rtg_b[idx] = student_rtg
        return teacher_traj_b, stacked_student_frames_b, teacher_attn_mask_b, student_attn_mask_b, teacher_time_steps_b, student_time_steps_b, target_student_actions_b, student_rtg_b

    def __getitem__(self, idx):
        """
        all trajectories are seen as one long stack, and we take the idx element of that giant stack, and prepend stack_size - 1 student
        frames from the same trajectory of element idx

        [f_0, f_0, f_0, f_1, f_2, ..., f_n], f_k = [s_k, a_k]

        if idx selected leads to a frame_idx of trajectory traj_idx thats not masked, wrap around to beginning frame of traj_idx

        """
        traj_idx = idx // self.max_student_length  # each trajectory is max_length long

        # wrap around, only allow samples of frames that have corresponding student data
        # disallow selecting the final frame as the current frame as there is nothing to predict then.
        frame_idx = (idx % self.max_student_length) % (int(self.student_trajectory_lengths[traj_idx]) - 1)


        teacher_traj = self.teacher_trajectories[traj_idx]
        teacher_traj_length = self.teacher_trajectory_lengths[traj_idx]
        student_traj = self.student_trajectories[traj_idx][self.student_attn_masks[traj_idx].bool()]
        if self.use_returns_to_go:
            student_rtg_full = self.student_rtgs[traj_idx][self.student_attn_masks[traj_idx].bool()]

        prepend_start = frame_idx - self.stack_size + 1
        stack_start = max(0, prepend_start)
        stacked_student_frames = student_traj[stack_start : frame_idx + 1]  # (stack_size, state_dim + act_dim)

        student_rtg = None
        if self.use_returns_to_go:
            student_rtg = student_rtg_full[stack_start : frame_idx + 1]
        student_attn_mask = torch.ones(self.stack_size, dtype=torch.bool)

        teacher_time_steps = torch.zeros(self.max_teacher_length, dtype=torch.long)
        teacher_time_steps[-teacher_traj_length:] = torch.arange(0, teacher_traj_length, dtype=torch.long)

        student_time_steps = torch.arange(stack_start, frame_idx + 1, dtype=torch.long) # (stack_size)

        if prepend_start < 0:
            # repeat beginning frames
            null_frame = torch.zeros(self.state_dim + self.act_dim)
            null_frame[self.state_dim:] = -10. # decision transformer does this. why?
            stacked_student_frames = torch.vstack(
                [null_frame.unsqueeze(0).repeat(-prepend_start, 1), stacked_student_frames]
            ) 
            student_time_steps = torch.hstack([torch.zeros(-prepend_start, dtype=torch.long), student_time_steps])

            if self.use_returns_to_go:
                student_rtg = torch.hstack([torch.zeros(-prepend_start), student_rtg])

            student_attn_mask[:-prepend_start] = False
        
        target_student_actions = student_traj[student_time_steps, self.state_dim:]
        if self.is_categorical:
            # convert from one hot back to indices for labels
            target_student_actions = np.argmax(target_student_actions, 1)
        
        return teacher_traj, stacked_student_frames, self.teacher_attn_masks[traj_idx], student_attn_mask, teacher_time_steps, student_time_steps, target_student_actions, student_rtg
