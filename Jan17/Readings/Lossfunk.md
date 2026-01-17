# Meta RL

Why Meta Rl ?
-To overcome the limitations of traditional RL algorithms in adapting to new tasks or environments quickly.
-To enable agents to learn new skills with less data and fewer interactions.
-To create more general-purpose AI systems that can solve a wider range of problems.
Meta RL --> Designing System that learn Rl algo Itself
STD RL vs Meta RL(Meta learning System)
Taxonomy of RL
Learn the policy -> classic metarl
learn the parts of the rl loop
learn the update rule end to end
use llms to generate codes

Accelerators for meta RL
-Accelerators#1: GPU based environments(gymnax,jax)pgx jumanji brax
-Accelerators#2: LLM guided evolution
Alpha Evolve (not open source)

Open Evolve ...

Classical meta rl

Break through

What are learned targets representing?

What is Meta Network?

Why human crafted algorithms fall short?
-The complexity of real-world problems often exceeds the assumptions made by human-designed algorithms.
-Human intuition can be limited in exploring the vast and complex search space of optimal algorithms.
-The development of new algorithms is a time-consuming and labor-intensive process.

use of LLMs +search

what is Alpha Evolve? AlphaEvolve is a system that uses large language models (LLMs) to generate and optimize reinforcement learning (RL) algorithms. It leverages the LLM's ability to understand and generate code to explore a vast space of potential algorithm designs, iteratively improving them based on performance metrics.

what is Jax? Jax is a high-performance numerical computing library developed by Google. It's designed for high-performance machine learning research, particularly for deep learning. Key features include:
-**Automatic Differentiation:** Jax can automatically differentiate native Python and NumPy functions.
-**JIT Compilation:** It uses XLA (Accelerated Linear Algebra) to compile Python and NumPy code into optimized kernels for GPUs and TPUs, leading to significant speedups.
-**Vectorization (vmap) and Parallelization (pmap):** Jax provides utilities for automatically vectorizing and parallelizing computations across multiple devices.

Isaac Gym: Isaac Gym is a high-performance GPU-accelerated simulation environment for robotics and reinforcement learning. It's built on NVIDIA's PhysX 5 physics engine and designed to simulate thousands of robotic environments in parallel on a single GPU. This parallelization significantly speeds up the data collection process for RL agents, allowing for much faster training times compared to traditional CPU-based simulators.

Exps at lossfunk
Gymnax -- gpu based
Gym-- cpu based

Attempt#1: Use evolulatary algos t invent new rl algos
Why this didn't work?

To conclude
-Instead of doing x, we should start building systems that do x is the main theme of meta rl.
