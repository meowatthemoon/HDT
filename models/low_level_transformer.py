import torch
import torch.nn as nn
import transformers

from models.gpt2 import GPT2Model


class LowLevelTransformer(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(self, state_size : int, action_size : int, d_model : int, max_length : int = None, max_ep_len : int = 4096, action_tanh=True, **kwargs):
        super().__init__()

        self.state_size : int = state_size
        self.action_size : int = action_size
        self.max_length : int = max_length
        self.d_model : int = d_model

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd = d_model,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, d_model)
        self.embed_subgoal = torch.nn.Linear(self.state_size, d_model)
        self.embed_state = torch.nn.Linear(self.state_size, d_model)
        self.embed_action = torch.nn.Linear(self.action_size, d_model)

        self.embed_ln = nn.LayerNorm(d_model)

        # note: we don't predict states or returns for the paper
        self.predict_action = nn.Sequential(
            *([nn.Linear(d_model, self.action_size)] + ([nn.Tanh()] if action_tanh else []))
        )



    def forward(self, states, actions, subgoals, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), device = states.device, dtype = torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        subgoal_embeddings = self.embed_subgoal(subgoals)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        subgoal_embeddings = subgoal_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (subgoal_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.d_model)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.d_model).permute(0, 2, 1, 3)

        # get predictions TODO
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return action_preds



    def get_action(self, states, actions, subgoals, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_size)
        actions = actions.reshape(1, -1, self.action_size)
        subgoals = subgoals.reshape(1, -1, self.state_size)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            subgoals = subgoals[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_size), device=states.device), states],
                dim=1).to(dtype=torch.float32)

            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.action_size),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)

            subgoals = torch.cat(
                [torch.zeros((subgoals.shape[0], self.max_length-subgoals.shape[1], self.state_size), device=subgoals.device), subgoals],
                dim=1).to(dtype=torch.float32)

            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        action_preds = self.forward(states, actions, subgoals, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]