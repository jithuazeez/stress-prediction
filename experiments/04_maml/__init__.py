"""
MAML (Model-Agnostic Meta-Learning) experiments.

Implements personalized stress prediction using meta-learning.
Each subject is treated as a separate task, and the model learns
to quickly adapt to new subjects with few samples.

References:
- Original paper: https://arxiv.org/pdf/1703.03400 (Finn et al., ICML 2017)
- Original code: https://github.com/cbfinn/maml
- Learn2learn library: https://github.com/learnables/learn2learn
"""

