# Solving Markov Decision Processes (MDPs)

## Value Iteration

The **Bellman Optimality Equation** gives a recursive definition of the optimal value function:

$$
V^*(s) = \max_{a} \sum_{s', r} P(s', r \mid s, a)\,[r + \gamma V^*(s')]
$$

### Key Idea
Solve iteratively using **Dynamic Programming**.

### Algorithm

- **Initialize:**

$$
V_0(s) = 0 \quad \forall s
$$

- **Iterate the Bellman update until convergence:**

$$
V_{k+1}(s) = \max_{a} \sum_{s', r} P(s', r \mid s, a)\,[r + \gamma V_k(s')]
$$

- **Stop when:**

$$
\max_s \left| V_{k+1}(s) - V_k(s) \right| < \epsilon
$$

### Intuition
- Information propagates outward from **terminal states**
- After enough iterations, all states converge to the correct value estimates

---

## Policy Iteration

An alternative approach that explicitly maintains a **policy**.

### Step 1: Policy Evaluation

Evaluate the value function for a **fixed policy** $\pi$.

- **Initialize:**

$$
V_0^{\pi}(s) = 0
$$

- **Iterative update:**

$$
V_{k+1}^{\pi}(s) =
\sum_{a} \pi(a \mid s)
\sum_{s', r} P(s', r \mid s, a)\,[r + \gamma V_k^{\pi}(s')]
$$

- Continue until convergence

---

### Step 2: Policy Improvement

Improve the policy using the current value estimates:

$$
\pi_{\text{new}}(s) =
\arg\max_{a} \sum_{s', r} P(s', r \mid s, a)\,[r + \gamma V^{\pi}(s')]
$$

Repeat **Policy Evaluation → Policy Improvement** until the policy stops changing.

---

## Value Iteration vs Policy Iteration

| Aspect | Value Iteration | Policy Iteration |
|------|----------------|------------------|
| Updates | Values directly | Policy + Values |
| Bellman Equation | Optimality equation | Expectation equation |
| Convergence | Slower (many iterations) | Faster outer loop |
| Computation | Simpler per iteration | More expensive per iteration |
| Inner Loop | No | Yes (policy evaluation) |

### Key Takeaway
- **Value Iteration**: simpler, more iterations
- **Policy Iteration**: fewer iterations, but each iteration is heavier

Both converge to the **optimal policy**.
