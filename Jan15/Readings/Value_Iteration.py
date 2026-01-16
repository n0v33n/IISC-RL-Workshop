import gymnasium as gym
import numpy as np
import imageio
import os

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs("value_iteration", exist_ok=True)

# -----------------------------
# Environment
# -----------------------------
env = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    render_mode="rgb_array"
)

n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.99
theta = 1e-6
max_iterations = 20   # limit for visualization

# -----------------------------
# Initialize value function
# -----------------------------
V = np.zeros(n_states)

# -----------------------------
# Helper: Extract policy from V
# -----------------------------
def extract_policy(V):
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            q = 0
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q += prob * (reward + gamma * V[next_state])
            q_values.append(q)

        policy[s] = np.argmax(q_values)

    return policy


# -----------------------------
# Helper: Run one episode & save GIF
# -----------------------------
def run_episode_and_save(policy, filename, max_steps=50):
    frames = []
    state, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        frames.append(env.render())
        action = policy[state]
        state, reward, terminated, truncated, _ = env.step(action)
        steps += 1

    frames.append(env.render())
    imageio.mimsave(filename, frames, fps=2)


# -----------------------------
# VALUE ITERATION
# -----------------------------
print("Starting Value Iteration...\n")

for iteration in range(1, max_iterations + 1):
    delta = 0

    for s in range(n_states):
        v = V[s]
        q_values = []

        for a in range(n_actions):
            q = 0
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q += prob * (reward + gamma * V[next_state])
            q_values.append(q)

        V[s] = max(q_values)
        delta = max(delta, abs(v - V[s]))

    # Extract policy after this iteration
    policy = extract_policy(V)

    print(f"Iteration {iteration}")
    print(V.reshape(4, 4))
    print(policy.reshape(4, 4))
    print("-" * 30)

    # Save episode GIF
    gif_path = f"value_iteration/iter_{iteration:02d}.gif"
    run_episode_and_save(policy, gif_path)

    if delta < theta:
        print("Converged!")
        break

# -----------------------------
# Save final policy GIF
# -----------------------------
run_episode_and_save(policy, "value_iteration/final.gif")

env.close()
print("\nAll GIFs saved in folder: value_iteration/")
