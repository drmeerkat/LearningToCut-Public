import os
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from collections import deque

from utils import layer_init, generate_log_path
from environments import CustomSyncVecEnv


class StateEmbedding(nn.Module):
    # This is the paper version embedding network
    # in 'Reinforcement Learning for Interger Programming: Learning to Cut'
    def __init__(self, n_vars, embed_dim) -> None:
        super().__init__()
        self.embed_layers = nn.LSTM(input_size=n_vars, hidden_size=embed_dim, num_layers=1, batch_first=True)
        # newly added!
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, cons, cuts):
        # output a tensor of shape (batch, seq len, features)
        embeds, _ = self.embed_layers(torch.concat((cons, cuts), dim=1))
        # normalize a tensor of (batch, features, seq len) and permute back
        normed_embeds = self.norm(embeds.permute(0, 2, 1)).permute(0, 2, 1)
        return normed_embeds[:, :cons.shape[1], :], normed_embeds[:, cons.shape[1]:, :]


class AttentionHead(nn.Module):
    # This is the paper version attentionhead network
    # in 'Reinforcement Learning for Interger Programming: Learning to Cut'
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, 128), std=0.01),
            nn.ReLU(),
        )

    def forward(self, cons_embed, cuts_embed):
        # of shape (batch, max # of cons, n_vars + 1), (batch, max # of cuts, n_vars + 1)
        # return a logits tensor of shape (batch, max # of cuts)
        cons_embed = self.model(cons_embed)
        cuts_embed = self.model(cuts_embed)
        logits = torch.mean(torch.matmul(cuts_embed, cons_embed.permute(0, 2, 1)), dim=-1)
        assert logits.shape[0] == cuts_embed.shape[0] and 'action logits should have the same batch dim as cuts!'
        assert logits.shape[1] == cuts_embed.shape[1] and 'action logits should have the dim of # of cuts!'
        return logits


class PPOAgent(nn.Module):
    """
    To enable batch training, this will take a zero-padded state input
    """
    def __init__(self, n_vars, embed_dim):
        super(PPOAgent, self).__init__()
        self.n_vars = n_vars

        self.actor_state_embed = StateEmbedding(n_vars, embed_dim)
        self.actor_att_head = AttentionHead(embed_dim)
        
        self.critic_state_embed = StateEmbedding(n_vars, embed_dim)
        self.critic = nn.Sequential(
            # since we concat the last hidden state of cons and cuts, 
            # feat dim is doubled
            layer_init(nn.Linear(embed_dim * 2, 128), std=0.01),
            nn.LeakyReLU(),
            layer_init(nn.Linear(128, 1), std=0.01)
        )

    def get_value(self, cons, cuts):
        # Only take the last hidden state and concat across feature dim
        cons_embed, cuts_embed = self.critic_state_embed(cons, cuts)
        x = torch.concat([cons_embed[:, -1, :].squeeze(1), cuts_embed[:, -1, :].squeeze(1)], dim=1)
        return self.critic(x)

    def get_action_and_value(self, cons, cuts, logit_mask, action=None):
        cons_embed, cuts_embed = self.actor_state_embed(cons, cuts)
        probs = self.actor_att_head(cons_embed, cuts_embed) + 1e-6
        masked_probs = torch.mul(probs, logit_mask)
        try:
            act_dist = Categorical(probs=masked_probs)
        except ValueError as e:
            import pickle
            with open('~/l2cut-error-probs', 'w') as f:
                pickle.dump(act_dist, f)
            print(act_dist.probs[0], act_dist.probs[0].sum())
            raise e
        if action is None:
            action = act_dist.sample()
        return action, act_dist.log_prob(action), act_dist.entropy(), self.get_value(cons, cuts)


def train(
        envs,
        evaluate_env, 
        agent: PPOAgent,
        args,
        device,
        timestamp,
        optimizer,
    ):
    '''
    The main training loop for the PPO agent
    '''

    # Store the batch policy rollouts for each train loop
    constraints = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space[0].shape).to(device)
    cuts = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space[1].shape).to(device)
    cuts_masks = torch.zeros((args.num_steps, args.num_envs) + (envs.single_observation_space[1].shape[0], )).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialize env input
    cur_cons, cur_cuts, cur_masks = envs.reset()
    cur_cons = torch.Tensor(cur_cons).to(device)
    cur_cuts = torch.Tensor(cur_cuts).to(device)
    cur_masks = torch.Tensor(cur_masks).to(device)
    cur_dones = torch.zeros(args.num_envs).to(device)

    # Total number of env steps been used (<= args.total_timesteps)
    global_steps = 0 
    # batch size = num_envs x num_steps
    num_updates = args.total_timesteps // args.batch_size

    # Tensorboard and checkpoints
    env_name = args.config
    writer = SummaryWriter(generate_log_path('tfboards_runs/' + timestamp + '_' + args.exp_name + '_' + env_name))
    check_point_path = generate_log_path('checkpoints/' + timestamp + '_' + args.exp_name + '_' + env_name)

    # Wandb rewards recording
    WINDOW = 100
    wan_train_rewards = deque([], WINDOW)
    wan_eva_rewards = deque([], WINDOW)

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            new_lr = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = new_lr
        
        # Collect policy rollouts
        for step in range(args.num_steps):
            global_steps += args.num_envs
            constraints[step] = cur_cons
            cuts[step] = cur_cuts
            cuts_masks[step] = cur_masks
            dones[step] = cur_dones
            with torch.no_grad():
                acts, lgp, _, v = agent.get_action_and_value(cur_cons, cur_cuts, cur_masks)
                values[step] = v.flatten()
                actions[step] = acts
                logprobs[step] = lgp

            next_obs, r, done, info = envs.step(acts.cpu().numpy())
            rewards[step] = torch.Tensor(r).to(device).flatten()

            # Update current observation
            cur_cons, cur_cuts, cur_masks = next_obs
            cur_cons = torch.Tensor(cur_cons).to(device)
            cur_cuts = torch.Tensor(cur_cuts).to(device)
            cur_masks = torch.Tensor(cur_masks).to(device)
            cur_dones = torch.Tensor(done).to(device)

            # Collect episode information
            for i, d in enumerate(done):
                if d:
                    # since there might be multiple envs "done", we will only record the first done episode
                    # print(f"global_step={global_steps}, episodic_return={info[i]['episode']['r']}")
                    writer.add_scalar("Train/cur_episodic_return", info[i]['episode']['r'], global_steps)
                    writer.add_scalar("Train/cur_episodic_length", info[i]['episode']['l'], global_steps)
                    if args.track:
                        wandb.log({f"Training Reward ({args.config})": info[i]['episode']['r']})
                        wan_train_rewards.append(info[i]['episode']['r'])
                        if len(wan_train_rewards) == wan_train_rewards.maxlen:
                            wandb.log({f"Training Reward Moving Avg ({args.config})": sum(wan_train_rewards)/wan_train_rewards.maxlen})
                    break

        # Estimate advantages for this batch rollouts
        with torch.no_grad():
            # the state value of one time step later after the end of this batch of rollouts
            after_batch_values = agent.get_value(cur_cons, cur_cuts).reshape(1, -1)
            # If an env is done, then next state value is 0 otherwise use returns/values
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nonterminal = 1.0 - cur_dones
                        next_values = after_batch_values
                    else:
                        nonterminal = 1.0 - dones[t+1]
                        next_values = values[t+1]
                    delta = rewards[t] + args.gamma * nonterminal * next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nonterminal * lastgaelam
                returns = advantages + values
            else:
                # the traditional way of estimating adv = return (like Q) - value
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nonterminal = 1.0 - cur_dones
                        next_returns = after_batch_values
                    else:
                        nonterminal = 1.0 - dones[t+1]
                        next_returns = returns[t+1]
                    # Q = E[discounted avg episode return]
                    returns[t] = rewards[t] + args.gamma * nonterminal * next_returns
                advantages = returns - values

        # Flatten the num_steps, num_envs dim of the whole batch
        cons_flatten = constraints.reshape((-1, ) + envs.single_observation_space[0].shape)
        cuts_flatten = cuts.reshape((-1, ) + envs.single_observation_space[1].shape)
        cuts_masks_flatten = cuts_masks.reshape((-1, ) + (envs.single_observation_space[1].shape[0], ))
        actions_flatten = actions.flatten()
        logprobs_flatten = logprobs.flatten()
        advantages_flatten = advantages.flatten()
        returns_flatten = returns.flatten()
        values_flatten = values.flatten()

        # Optimize the agent!
        batch_indices = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # Reuse the same rollout batch for multiple updates
            # By further splitting the rollout batch into random minibatches 
            np.random.shuffle(batch_indices)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_indices = batch_indices[start:end]

                # Get new values, logprobs with the updated agent and compare with batched data
                _, new_logprobs, new_entropy, new_values = agent.get_action_and_value(
                    cons_flatten[minibatch_indices], 
                    cuts_flatten[minibatch_indices], 
                    cuts_masks_flatten[minibatch_indices],
                    actions_flatten.long()[minibatch_indices]
                )
                logratio = new_logprobs - logprobs_flatten[minibatch_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    # debug vars: how far away the new policy is from the old one?
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # debug vars: how often the pg loss clip is triggered?
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Advantage normalization
                minibatch_advantages = advantages_flatten[minibatch_indices]
                if args.norm_adv:
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                # PPO policy loss (clipped version of PPO objective)
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # PPO value loss
                new_values = new_values.flatten()
                v_loss_unclipped = (new_values - returns_flatten[minibatch_indices]) ** 2
                if args.clip_vloss:
                    v_clipped = values_flatten[minibatch_indices] + torch.clamp(
                        new_values - values_flatten[minibatch_indices],
                        -args.clip_coef,
                        args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - returns_flatten[minibatch_indices]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_clipped, v_loss_unclipped).mean()
                else:
                    v_loss = 0.5 * v_loss_unclipped.mean()

                # Entropy loss (to encourage exploration, maximize entropy)
                entropy_loss = new_entropy.mean()

                # Final loss
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # Optimize the param
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Minibatch level early stopping
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

        # debug vars: how accurate the value function is?
        y_pred, y_true = values_flatten.cpu().numpy(), returns_flatten.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Note that most attributes recorded here are only from the last minibatch in the last update
        writer.add_scalar("Train/learning_rate", optimizer.param_groups[0]["lr"], global_steps)
        writer.add_scalar("Train/value_loss", v_loss.item(), global_steps)
        writer.add_scalar("Train/policy_loss", pg_loss.item(), global_steps)
        writer.add_scalar("Train/entropy", entropy_loss.item(), global_steps)
        writer.add_scalar("Train/approx_kl", approx_kl.item(), global_steps)
        writer.add_scalar("Train/clipfrac", np.mean(clipfracs), global_steps)
        writer.add_scalar("Train/explained_variance", explained_var, global_steps)
        writer.add_scalar("Train/progress", update*1.0/num_updates, global_steps)

        if update % args.log_interval == 0:
            eva_return = evaluate(evaluate_env, agent, args, device, ep=1)
            torch.save(agent.state_dict(), os.path.join(check_point_path, f'{update}.pt'))
            writer.add_scalar("Evaluate/episode_return", eva_return, global_steps)
            print(f"{global_steps}/{args.total_timesteps} - evaluate return: {eva_return}")
            if args.track:
                wan_eva_rewards.append(eva_return)
                wandb.log({f"Evaluation Reward ({args.config})": eva_return})
                if len(wan_eva_rewards) == wan_eva_rewards.maxlen:
                    wandb.log({f"Evaluation Reward Moving Avg ({args.config})": sum(wan_eva_rewards)/wan_eva_rewards.maxlen})

        if args.debug:
            break

    torch.save(agent.state_dict(), os.path.join(check_point_path, f'final.pt'))
    envs.close()
    writer.close()


def evaluate(env: CustomSyncVecEnv, agent: PPOAgent, args, device, ep):
    with torch.no_grad():
        total_reward = 0
        for i in tqdm(range(ep), desc='Eva Progress:'):
            state = env.reset()
            done = False
            while not done:
                cur_cons, cur_cuts, cur_masks = state
                cur_cons = torch.Tensor(cur_cons).to(device).unsqueeze(0)
                cur_cuts = torch.Tensor(cur_cuts).to(device).unsqueeze(0)
                cur_masks = torch.Tensor(cur_masks).to(device).unsqueeze(0)
                action, _, _, value = agent.get_action_and_value(cur_cons, cur_cuts, cur_masks)
                nxt_state, reward, done, info = env.step(action.item())
                if args.debug:
                    print('state:\n', state)
                    print('action:', action.item())
                    print('state value:', value.item())
                    print('reward:', reward)
                    print('done:', done)
                    print("-----------------")
                state = nxt_state
            total_reward += info['episode']['r']
    return total_reward*1.0/ep
