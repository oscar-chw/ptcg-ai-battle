# References and what they support

1. Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.** Proceedings of the 36th International Conference on Machine Learning, PMLR 97:3744–3753, 2019. [Official paper record](https://proceedings.mlr.press/v97/lee19d.html). Supports attention-based set interactions and invariant pooling. The PTCG instance-pointer mapping must be checked separately.

2. Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.** Proceedings of AISTATS, PMLR 15:627–635, 2011. [Official paper record](https://proceedings.mlr.press/v15/ross11a.html). Supports analysis of learner-induced distribution shift and dataset aggregation under its stated assumptions. Winning-only cloning is not DAgger.

3. Peter I. Cowling, Edward J. Powley, and Daniel Whitehouse. **Information Set Monte Carlo Tree Search.** IEEE Transactions on Computational Intelligence and AI in Games 4(2):120–143, 2012. DOI 10.1109/TCIAIG.2012.2200894. [Author-institution record](https://pure.york.ac.uk/portal/en/publications/information-set-monte-carlo-tree-search/). Supports information-set search designs for hidden-information games. It does not certify the strength of the project's implementation.

4. Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. **FiLM: Visual Reasoning with a General Conditioning Layer.** Proceedings of AAAI 32(1), 2018. DOI 10.1609/aaai.v32i1.11671. [Official paper record](https://ojs.aaai.org/index.php/AAAI/article/view/11671). Supports feature-wise affine conditioning. The bundled local v2 comparison does not isolate FiLM's individual effect.

5. Thomas Anthony, Zheng Tian, and David Barber. **Thinking Fast and Slow with Deep Learning and Tree Search.** Advances in Neural Information Processing Systems 30, 2017. [Official paper record](https://papers.nips.cc/paper_files/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html). Supports expert iteration combining planning and an apprentice model. Its successful application here remains proposed.

Additional works discussed in the campaign's longer technical appendix:

6. Ivo Danihelka, Arthur Guez, Julian Schrittwieser, and David Silver. **Policy improvement by planning with Gumbel.** ICLR, 2022. [Official record](https://openreview.net/forum?id=bERaNdoegnO). Its policy-improvement premise requires correctly evaluated action values; an arbitrary biased critic does not inherit the guarantee.

7. Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. **On Calibration of Modern Neural Networks.** ICML, PMLR 70:1321–1330, 2017. [Official paper record](https://proceedings.mlr.press/v70/guo17a.html). Supports confidence-calibration analysis. The report's constant-Brier identity is elementary algebra applied to local class frequencies, not a theorem attributed to this paper.

Competition sources: [Strategy overview and rubric](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/overview), [Strategy rules](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/rules), and [Simulation leaderboard](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard). Live competition facts were checked on 13 September 2026.
