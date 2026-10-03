# Mechanics, Learning, and Reliability in PTCG

*Deck-conditioned imitation, structured legal actions, and an experimental study of what makes learned play trustworthy*

## 1. Contribution and submitted agent

PTCG asks an agent to reason about resources whose value changes with identity, location and timing. An Energy card in discard can be tomorrow's attacker; a healing action can prevent a knockout while disabling today's attack. The learning problem is therefore richer than recognizing cards or selecting the largest damage number.

We built deck-specialist imitation agents and investigated belief inference, value learning, search and simulator acceleration. Our engineering contribution is a connected methodology: **represent the game's concrete relationships, preserve that representation through deployment, and test strategic claims in controlled games**. The campaign produced an implemented legal-option policy, useful experimental infrastructure, and negative results that delimit the next research step. These are separate development branches, not an assertion that every mechanism shipped together.

Our final Simulation standing was **2043/6807, score 709.1**, with **BEST1_fixed.tar.gz**. Its submission record specifies a Mega Lucario specialist trained on **400 winning games from four demonstrators**: d512, 12 layers, eight heads, FFN1365 and 269 numeric features. The selected epoch-12 checkpoint recorded **58.16% contested validation accuracy**. These are recorded configuration and validation facts, not independently reconstructed final-archive results. Earlier development tests below do not measure BEST1's win rate. We did not recover a controlled final-agent evaluation across seats, repeated matches and matchups; those robustness criteria remain unestablished.

## 2. Deck design as a resource-flow problem

**Mega Lucario ex, our final submission's archetype.** Aura Jab deals 130 for one Fighting energy and accelerates up to three Basic Fighting **from discard to Benched Pokémon**. Mega Brave deals 270 for two Fighting, but that Pokémon cannot use **Mega Brave** next turn. The lower-damage attack can secure a knockout while preparing a successor; the higher-damage attack can cross an otherwise unreachable knockout threshold. Their ranking depends on the board and future resources.

We recovered cohort `ed7c14bab680` from two independently hash-matching replay deck actions: **16 Pokémon, 31 Trainers, 13 Fighting Energy**. The list contains Riolu/Mega Lucario at 3/4, Makuhita/Hariyama at 2/2, Lunatone/Solrock at 2/3, and four each of Ultra Ball, Fighting Gong and Poké Pad. Its architecture is coherent: Lunatone, with Solrock in play, discards Fighting to draw three once globally per turn; Ultra Ball's discard cost can also supply Aura Jab. Fighting Gong finds Basic Fighting Pokémon or Energy, whereas Poké Pad excludes Rule Box Pokémon; Ultra Ball can find Mega Lucario itself. Collapsing these items into one search feature loses the distinction that only Ultra Ball can fetch Mega Lucario.

Hariyama's evolution-triggered gust can preserve the Supporter choice for Judge, Lillie's Determination or healing. Wally's Compassion heals a Mega Evolution ex fully but returns attached Energy to hand when damage is healed. Hero's Cape adds durability; Premium Power Pro changes damage thresholds. The model must price survival against attack readiness, not reward HP restoration in isolation. We explain the observed list without claiming its card counts are optimal. Its association with the final archive comes from submission metadata; the final archive was unavailable for independent reconstruction.

**Marnie and Festival exposed further representation requirements.** Our preserved Marnie specialist has ten Dark Energy, four Munkidori, two Froslass and a 4/3/3 Marnie's evolution line. Grimmsnarl's Punk Up supplies **Marnie's Pokémon**, excluding Munkidori. Ordinary attachment and evolution acceleration are consequently different resource channels. Froslass damages Ability holders on both boards at Checkup, except Froslass. A powered Munkidori transfers at most three existing counters from **one donor to one target**, once per turn per copy. Pooling damage across donors overstates a single activation's capacity.

In Festival, Thwackey's tutor needs a Festival Lead Active; Festival Grounds separately enables Dipplin's second attack. Promotion can therefore unlock a search before finding the Stadium. This supplies a concrete counterexample to treating a turn as an unordered action set: `b` can be illegal before `a`, yet legal afterward. We execute one response, observe the resulting state, and reason over the new legal menu.

## 3. From relational state to a variable-action policy

The environment combines hidden hands, deck order and Prizes with public board state and a sequence of resolution prompts. We treat the competition simulator as the operational authority for transitions and legal options. Features must use information available to the player; privileged simulator state is not an inference-time input.

For observation encoding `o` and real legal options `A(o)`, the single-choice policy is

`pi(a|o) = exp(z_a) / sum_{b in A(o)} exp(z_b)`.

The inspected historical neural implementation uses typed entity and option tokens, card identity embeddings, numeric projections, attention and explicit option-to-entity pointers. Pooled state and option representations produce the scores. Set attention [1] shares statistical structure across variable boards; pointers retain **which instance** an action affects. Two identical Pokémon can differ in damage, attached energy and successor role. Legal feasibility alone does not encode these opportunity costs.

The useful symmetry is entity permutation with consistent pointer and relation remapping, not indifference to temporal order. Multiple-selection prompts additionally require cardinality and combination constraints; individually valid indices do not guarantee a valid complete response. The inspected policy executes greedily and does not use its value head for search. We do not relabel imitation as completed reinforcement learning.

Behavioral cloning minimizes demonstrated-action cross-entropy. One preserved Marnie corpus contains **89,048 decisions from 926 games**, split by whole games into **741/185 training/validation games**, with zero recorded build failures. This prevents adjacent frames from one game crossing the split, but leaves dependence within games and possible demonstrator/matchup overlap. We distinguish contested-choice accuracy from forced-choice-inflated overall accuracy.

Winning-only supervision concentrates successful trajectories but retains mistakes and changes the sampling distribution. On-policy states can differ from demonstrations, as dataset-aggregation research explains [2]. Accordingly, imitation accuracy is a diagnostic; neither a win-rate estimator nor a proof that the student cannot exceed its teacher. Demonstrator and opponent holdouts remain necessary generalization tests.

## 4. Three contracts that make experiments interpretable

**Representation invariance.** With `m` zero-key, zero-value padding pairs, unmasked attention has denominator `sum_j exp(qk_j) + m exp(0)`: padding changes real-token contributions despite contributing no values. If padded keys are excluded at every layer, padded queries are excluded from pooling, and other operations preserve the same real-token structure, adding padding preserves real outputs in exact arithmetic. This follows layer by layer; floating-point kernels and the actual decoder still need testing.

A documented trained-weight probe on seeded synthetic tokens changed width 160 to 192: **37/300** argmax flips without masking versus **0/300** with masking. This supports the mechanism on that probe, not a deployed-board error rate. The practical consequence is that encoder shape, masks, weights and decoder jointly define the deployed function. Model size or offline accuracy cannot rescue a comparison between mismatched functions.

**Value semantics.** For multiclass sum Brier loss, the class-frequency constant predictor has expected loss `1 - sum_c p_c^2`. Our preserved validation artifact yields **0.487022** across **17,392 rows/185 games**. A “Brier below 0.5” gate can therefore admit zero skill. Outcome prediction under demonstrated play also does not identify counterfactual action values. A planner must validate both predictive skill and action ranking.

Continuation values must retain a fixed root player's utility. A sign change converts between player perspectives; it must not occur after every API prompt, since multiple target and resolution choices can belong to one strategic turn. This is a consistency requirement for proposed search, not a claim that the deployed imitation policy performs it.

**Measurement identity.** Freeze candidate, deck, opponent, engine, seed schedule, actual seat and budget. Identical search-enabled arms sharing seeded inputs agreed on **132/200** outcomes; search-disabled repetition agreed on **200/200**. Common randomness does not make wall-clock search deterministic. Fixed expansion counts test semantic equivalence; deployment time budgets test practical strength. These are distinct experiments.

## 5. Results and what they permit us to conclude

The following are preserved development experiments, not measurements of BEST1.

| Development test | Result | Decision |
|---|---|---|
| Corrected Marnie rule/search baseline vs Alakazam | 198/400; 49.50% [44.63,54.38] | Retire the invalid historical comparator. |
| ISMCTS vs same-deck baseline | 111/400; 27.75% [23.59,32.33] | Reject this candidate. |
| ISMCTS cross-deck battery | 108/400; 27.00% [22.88,31.55] | Do not promote. |
| Self-play instrumentation | 32/32 complete; 5,513 records; zero illegal decisions | Mechanics pass; learning improvement untested. |

Intervals are nominal 95% Wilson intervals for the recorded samples, not uncertainty over all opponents. The corrected baseline won **116/200 first-seat versus 82/200 second-seat**: a 17-point gap that makes seat balance essential. The rollout gate detected **5/5 deliberate mutations and 4/4 corrupted decisions**. These controls test whether selected failures are observable; they do not establish universal correctness.

ISMCTS [3] failed in our evaluated implementation, not as a general research direction. Search quality depends on beliefs, transition fidelity, rollout evaluation and budget. Sophistication without those controls can amplify error.

The systems branch produced a useful bounded gain: a preserved official-engine optimization measured **1.497701x incremental speedup** over the preceding implementation in rotating fixed-work batches. Fifty paired games matched every outcome and **all 9,844 moves**. This result applies to the tested macOS build and workload. It establishes a bounded engineering gain under a semantic oracle; strategic strength requires separate evaluation.

## 6. Research agenda with measurable expectations

An implemented research variant combines card-family FiLM [4], multiple pooling seeds and standardized numerics. A documented one-seed, two-epoch comparison improved contested accuracy **72.81% to 73.71%**. Because components changed together, attribution is unresolved; arena improvement is unestablished. The next experiment is a factorial, multi-seed comparison with fixed serving semantics and opponent holdouts. Expected output: component effects and uncertainty, not a promised rating gain.

A tactical planner should begin with short, verifiable continuations and hidden-state samples consistent with public evidence. Its acceptance test is improved action ranking on verified cases followed by budget-matched games. A sharper posterior alone is insufficient.

Expert iteration [5] would then use a demonstrably stronger search operator to generate alternative-action targets, train an apprentice, and evaluate fresh matchups against a frozen league. Our rollout infrastructure supports this experiment; the learning loop has not yet demonstrated improvement. Scaling comes after the expert's advantage is established.

The transferable lesson is that game understanding, mathematical contracts and systems verification belong in the same learning pipeline. Our competitive result is modest; the contribution is a precise account of which mechanisms were implemented, which hypotheses survived measurement, and what evidence would justify the next improvement.

## References

[1] Lee, J. et al. (2019). [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://proceedings.mlr.press/v97/lee19d.html). ICML.

[2] Ross, S., Gordon, G. and Bagnell, D. (2011). [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html). AISTATS.

[3] Cowling, P., Powley, E. and Whitehouse, D. (2012). [Information Set Monte Carlo Tree Search](https://pure.york.ac.uk/portal/en/publications/information-set-monte-carlo-tree-search/). IEEE TCIAIG.

[4] Perez, E. et al. (2018). [FiLM: Visual Reasoning with a General Conditioning Layer](https://ojs.aaai.org/index.php/AAAI/article/view/11671). AAAI.

[5] Anthony, T., Tian, Z. and Barber, D. (2017). [Thinking Fast and Slow with Deep Learning and Tree Search](https://papers.nips.cc/paper_files/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html). Advances in Neural Information Processing Systems 30.


## Supporting artifacts

See [methods and conditional proofs](METHODS.md), [aggregate results](../results/results.csv), [reproducibility](REPRODUCIBILITY.md), [full references](REFERENCES.md) and [limitations](LIMITATIONS.md). The original simulator, replays and final checkpoint are not redistributed in this package.
