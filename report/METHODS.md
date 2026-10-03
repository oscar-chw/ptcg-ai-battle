# Mathematical and experimental methods

These arguments explain design requirements. They are conditional statements and standard derivations, not claims of a new general learning theorem or proof that a neural policy is optimal.

## 1. Observation, actions and utility

Let the environment be an episodic stochastic game. The full state may contain hidden hands, Prize cards and deck order. A player receives an observation history h and a response schema. An implementable policy uses only information available to that player.

For a single-choice prompt, let A(o) be the set of real legal options. Given model scores z, the masked policy is

$$\pi(a\mid o)=\frac{\exp(z_a)}{\sum_{b\in A(o)}\exp(z_b)}.$$

For multiple selections, a feasible family of responses replaces A(o). That family includes cardinality, uniqueness and joint compatibility constraints. Validating every returned index individually does not validate the complete response.

The inspected historical architecture uses entity and option embeddings, attention, numeric projections and pointers from options to entity instances. Entity permutation requires the pointers and relations to be remapped consistently. It does not imply that temporal order can be discarded.

## 2. A constructive counterexample to unordered turns

Consider our main-action phase with a nonempty deck, an unused Thwackey, a non-Festival Active, a legal switch or promotion into a Festival Lead Active, and no suppressing effect. Let a be an action that completes that change of Active, and let b activate Thwackey's tutor.

$$b\notin A(s),\qquad b\in A(T(s,a)).$$

The forward sequence may be legal while the reverse sequence is undefined. Thwackey's tutor requires a Festival Lead Active; Festival Grounds is separately relevant to the second attack. This proves that a representation which always treats whole-turn actions as an unordered bag discards a necessary dependency. It does not prove that any particular search method improves win rate.

Reobserving after a response also matters when the action reveals information or changes the next legal menu. A plan based on the previous menu can become invalid even without an opponent turn.

## 3. Why zero padding is not neutral

For query q and real key-value pairs (k_j,v_j), add m padding pairs with zero keys and zero values. Unmasked attention becomes

$$\frac{\sum_j\exp(q^\top k_j)v_j}{\sum_j\exp(q^\top k_j)+m}.$$

The zero padding contributes nothing to the numerator but changes the denominator. A residual connection does not generally cancel this effect.

**Conditional invariance.** Assume padded keys are excluded in every attention layer; padded query outputs are excluded from pooling; other operations are tokenwise or preserve the same semantic structure; and pointers retain their intended entity identities. In exact arithmetic, adding padding leaves real-token outputs unchanged.

The proof is inductive. A real query sees the same real keys and values with the same normalizer. Subsequent tokenwise operations preserve equality. Repeating the argument through the network preserves real-token representations; consistent pooling and pointers preserve final scores.

Finite-precision kernels, shape-dependent reduction order, truncation and decoder ties require separate tests. The historical 37/300 versus 0/300 argmax result came from a documented synthetic-token probe with trained weights. The derivation above does not reproduce that trained-model experiment.

## 4. The constant-predictor Brier baseline

Let Y be a one-hot outcome and p its class-frequency distribution. Under multiclass **sum** Brier loss,

$$\mathbb{E}\left[\sum_c(p_c-Y_c)^2\right]=1-\sum_c p_c^2.$$

Expanding the square gives the identity because the expected value of Y_c is p_c and the class probabilities sum to one. For the recorded validation-row distribution, the baseline is approximately 0.487022. The result is data-dependent; it is not a universal threshold for all WDL datasets.

The reference is fitted retrospectively to validation prevalence for a descriptive no-skill comparison. A deployable constant predictor should estimate its frequencies from training or an independent calibration set. Repeated outcome labels within the same game are correlated, so interval estimation should respect game-level clustering. The validation rows are not 17,392 independent games.

Brier skill score is

$$\mathrm{BSS}=1-\frac{\mathrm{Brier}_{model}}{\mathrm{Brier}_{reference}}.$$

Skill relative to this baseline, confidence calibration, action ranking and improved play are different properties. A well-calibrated outcome predictor under demonstrated play does not identify counterfactual action values without additional assumptions or intervention data.

## 5. Fixed-root value orientation

Let r be a fixed root player, with terminal utility u_r in {-1,0,+1}. At a chosen next evaluation boundary tau,

$$Q_r(h,a)=\mathbb{E}[V_r(H_\tau)\mid h,a].$$

Chance, belief sampling and the specified continuation policies are part of the expectation. At termination, V_r=u_r. A value head declared from another player's perspective must be converted consistently.

Sign changes convert utility perspective. They do not automatically occur after every API prompt: donor selection, targeting, tutoring and attack resolution can all occur within one strategic turn. Negamax-style alternation requires an appropriate player-changing boundary. This is a contract for proposed search; the inspected historical imitation executor does not use a value head for planning.

## 6. Comparisons, common randomness and uncertainty

Bind every match comparison to candidate, deck, baseline, opponents, engine, seed schedule, actual starting seats, time or expansion budget, timeout policy and terminal result extraction. Incomplete shards and failed games must remain visible.

The recorded baseline's first-seat win rate was 116/200 and second-seat win rate 82/200. Their difference is 17 percentage points. This descriptive gap motivates seat-balanced testing; it does not isolate every causal source of first-player advantage.

For paired outcomes X and Y,

$$\mathrm{Var}(X-Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)-2\mathrm{Cov}(X,Y).$$

Common random numbers help when they induce favorable covariance. They do not imply deterministic outcomes under wall-clock search: identical search-enabled arms agreed on 132/200 outcomes, compared with 200/200 when search was disabled. Fixed expansion budgets are useful for semantic tests, while actual time budgets remain necessary for deployment comparisons.

The report's Wilson intervals are nominal Bernoulli intervals for the recorded samples. They do not account automatically for repeated opponents, paired seeds or hierarchical matchup structure. A game-level paired bootstrap or opponent-stratified analysis is preferable when appropriate raw records are available. Overlap of two separate intervals is not a test of equality.

## 7. A defensible next experiment

For a new representation or planner, freeze an exact baseline, define the smallest causal change, validate feature and serving semantics, and evaluate multiple seeds and held-out opponents. Use a separate confirmation set after exploratory model selection. The current corpus and final checkpoint provenance do not support inventing a final-agent optimizer, learning-rate schedule or cross-matchup result.

A counterfactual expert-iteration program needs a genuinely stronger search operator before using its choices as improved labels. A GPU simulator needs official-oracle, CPU and device semantic parity before throughput is treated as useful. An instrumentation pass is a prerequisite for these experiments, not evidence that they have succeeded.

See the [research report](REPORT.md) for primary literature and the [reproducibility capsule](REPRODUCIBILITY.md) for the scope of executable checks.
