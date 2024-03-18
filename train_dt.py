import argparse
import json
import os

import d4rl
import gym
import numpy as np
import torch
from torch.nn import MSELoss

from dataset import Dataset
from models.decision_transformer import DecisionTransformer

RESULTS_PATH = "./Results/DecisionTransformer"

def eval_episodes(env, model : DecisionTransformer, target_rew, num_eval_episodes, max_ep_len, scale, action_range : float, action_size : int, state_size : int, state_mean : float, state_std : float, sequence_length : int,device):
    model.eval()
    model.to(device = device)

    state_mean = torch.from_numpy(state_mean).to(device = device)
    state_std = torch.from_numpy(state_std).to(device = device)
    soa = np.zeros((1, action_size), dtype = np.float32)

    episode_returns, episode_lengths = [], []
    for episode_i in range(num_eval_episodes):
        state = env.reset()
        actions = torch.from_numpy(soa).reshape(1, 1, action_size).to(device = device, dtype = torch.float32)
        states = torch.from_numpy(state).reshape(1, 1, state_size).to(device = device, dtype = torch.float32)
        timesteps = torch.tensor(0).reshape(1, 1).to(device = device, dtype = torch.long)
        target_return = target_rew / scale
        ep_return = target_return
        target_return = torch.tensor(ep_return, device = device, dtype = torch.float32).reshape(1, 1, 1)

        episode_return, episode_length = 0, 0
        for episode_t in range(max_ep_len):

            state_seq = (states.to(dtype = torch.float32) - state_mean) / state_std
            state_seq = state_seq.to(device = device, dtype = torch.float32)

            action_preds = model.forward(
                states =  state_seq,
                actions = actions.to(device = device, dtype = torch.float32),
                timesteps = timesteps.to(device = device, dtype = torch.long),
                returns_to_go = target_return.to(device = device, dtype = torch.float32)
            )
            action = action_preds[0][-1] * action_range
            actions = torch.cat([actions, action.reshape(1, 1, action_size)], dim = 1)
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)

            cur_state = torch.from_numpy(state).to(device = device).reshape(1, 1, state_size)
            states = torch.cat([states, cur_state], dim = 1)

            pred_return = target_return[0][-1] - (reward/scale)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1, 1)], dim=1)

            episode_return += reward
            episode_length += 1
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length)], dim=1)

            states = states[:,-sequence_length:]
            actions = actions[:,-sequence_length:]
            target_return = target_return[:,-sequence_length:]
            timesteps = timesteps[:,-sequence_length:]

            if done:
                break
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
    return episode_returns, episode_lengths

    

def train(env_name : str, dataset_name : str, batch_size : int, d_model : int, eval_every : int, iterations : int, lr : float, num_eval_episodes : int, n_head : int, n_layer : int, sequence_length : int, weight_decay : float, warmup_steps : int, dropout : float):
    experiment_name = f'{env_name}_{dataset_name}_E{iterations}_D{d_model}_L{n_layer}_H{n_head}_K{sequence_length}'
    os.makedirs(RESULTS_PATH, exist_ok = True)

    dtype = torch.float32
    dataset : Dataset = Dataset(env_name = env_name, dataset = dataset_name, scale = 1000, dtype = dtype)
    action_range : float = dataset.action_range
    action_size : int = dataset.action_size
    state_size : int = dataset.state_size

    env = gym.make(dataset.full_env_name)

    model = DecisionTransformer(
            state_size = state_size,
            action_size = action_size,
            max_length = sequence_length,
            d_model = d_model,
            n_layer = n_layer,
            n_head = n_head,
            n_inner = 4 * d_model,
            activation_function = 'relu',
            n_positions = 1024,
            resid_pdrop = dropout,
            attn_pdrop = dropout
        )
    model = model.to(device = dataset.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))

    # Loss Function
    loss_fn = MSELoss(reduction = 'none').to(device = dataset.device)

    # Loop
    train_losses, val_10000_rewards_mean, val_10000_rewards_std, val_10000_lengths_mean, val_10000_lengths_std, val_max_rewards_mean, val_max_rewards_std, val_max_lengths_mean, val_max_lengths_std, val_half_rewards_mean, val_half_rewards_std, val_half_lengths_mean, val_half_lengths_std = [], [], [], [], [], [], [], [], [], [], [], [], []
    for iteration in range(iterations):
        model.train()

        states, actions, label_actions, _, _, rtg, timesteps, mask = dataset.get_batch(batch_size = batch_size, sequence_length = sequence_length)

        action_preds = model.forward(
            states = states, actions = actions, returns_to_go = rtg, timesteps = timesteps, attention_mask = mask
        )
        action_preds = action_preds * action_range

        loss = loss_fn(action_preds, label_actions)
        loss_mask = mask.reshape(batch_size, sequence_length) 
        loss = loss[loss_mask > 0].mean() # (Batch, Seq_len)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.detach().cpu().item())

        # Eval
        if (iteration + 1) % eval_every == 0:
            rewards, lengths = eval_episodes(target_rew = 10000, max_ep_len = dataset.max_ep_len, model = model, num_eval_episodes = num_eval_episodes, env = env, scale = dataset.scale, state_mean = dataset.state_mean, state_std = dataset.state_std, action_range = action_range, action_size = action_size, state_size = state_size, sequence_length = sequence_length, device = dataset.device)
            val_10000_rewards_mean.append(np.mean(rewards))
            val_10000_rewards_std.append(np.std(rewards))
            val_10000_lengths_mean.append(np.mean(lengths))
            val_10000_lengths_std.append(np.std(lengths))
            
            rewards, lengths = eval_episodes(target_rew = dataset.max_ep_rew, max_ep_len = dataset.max_ep_len, model = model, num_eval_episodes = num_eval_episodes, env = env, scale = dataset.scale, state_mean = dataset.state_mean, state_std = dataset.state_std, action_range = action_range, action_size = action_size, state_size = state_size, sequence_length = sequence_length, device = dataset.device)
            val_max_rewards_mean.append(np.mean(rewards))
            val_max_rewards_std.append(np.std(rewards))
            val_max_lengths_mean.append(np.mean(lengths))
            val_max_lengths_std.append(np.std(lengths))
            
            rewards, lengths = eval_episodes(target_rew = dataset.max_ep_rew / 2, max_ep_len = dataset.max_ep_len, model = model, num_eval_episodes = num_eval_episodes, env = env, scale = dataset.scale, state_mean = dataset.state_mean, state_std = dataset.state_std, action_range = action_range, action_size = action_size, state_size = state_size, sequence_length = sequence_length, device = dataset.device)
            val_half_rewards_mean.append(np.mean(rewards))
            val_half_rewards_std.append(np.std(rewards))
            val_half_lengths_mean.append(np.mean(lengths))
            val_half_lengths_std.append(np.std(lengths))

            print(f"{env_name}-{dataset_name} | {iteration + 1} / {iterations} | 10K = {val_10000_rewards_mean[-1]:.2f} | Max = {val_max_rewards_mean[-1]:.2f} | Half = {val_half_rewards_mean[-1]:.2f} | Total = {dataset.max_ep_rew}")

    # Save Results
    with open(os.path.join(RESULTS_PATH, f"{experiment_name}.json"), "w") as f:
        json.dump({
            "train_losses" : train_losses,
            "val_10000_rewards_mean" : val_10000_rewards_mean,
            "val_10000_rewards_std" : val_10000_rewards_std,
            "val_10000_lengths_mean" : val_10000_lengths_mean,
            "val_10000_lengths_std" : val_10000_lengths_std,
            "val_max_rewards_mean" : val_max_rewards_mean,
            "val_max_rewards_std" : val_max_rewards_std,
            "val_max_lengths_mean" : val_max_lengths_mean,
            "val_max_lengths_std" : val_max_lengths_std,
            "val_half_rewards_mean" : val_half_rewards_mean,
            "val_half_rewards_std" : val_half_rewards_std,
            "val_half_lengths_mean" : val_half_lengths_mean,
            "val_half_lengths_std" : val_half_lengths_std,
        },fp = f, indent = 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type = str)
    parser.add_argument("--dataset", type = str)
    parser.add_argument("--K", type = int)
    parser.add_argument("--d_model", type = int)
    parser.add_argument("--n_layer", type = int)
    parser.add_argument("--n_head", type = int)
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--eval_every", type = int, default = 1000)
    parser.add_argument("--iterations", type = int, default = 100000)
    parser.add_argument("--num_eval_episodes", type = int, default = 10)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--weight_decay", type = float, default = 1e-4)
    parser.add_argument("--warmup_steps", type = int, default = 10000)
    parser.add_argument("--dropout", type = float, default = 0.1)
    args = parser.parse_args()

    train(
        env_name = args.env_name, 
        dataset_name = args.dataset, 
        batch_size = args.batch_size, 
        d_model = args.d_model, 
        eval_every = args.eval_every, 
        iterations = args.iterations, 
        lr = args.lr, 
        num_eval_episodes = args.num_eval_episodes, 
        n_layer = args.n_layer, 
        n_head = args.n_head,
        sequence_length = args.K, 
        weight_decay = args.weight_decay, 
        warmup_steps = args.warmup_steps,
        dropout = args.dropout
    )