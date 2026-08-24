# Binary-option market makers

This folder contains four self-contained Python market-making strategies for binary options on two company underlyings and the Fed funds rate.  A binary option pays `1` when its observable finishes at or above its strike, and `0` otherwise.  Each script includes the market data types, statistical calibration, pricing model, quoting logic, position accounting, and fill-or-kill (FOK) decision rule needed by the competition interface.

## Models

| File | Style | Trading posture |
| --- | --- | --- |
| `aggressive_market_maker.py` | Aggressive | Highest displayed size and capital deployment. |
| `hybrid_market_maker.py` | Hybrid | Same risk-aware engine as the aggressive model, with lower deployment limits and a larger reserve. |
| `passive_market_maker.py` | Passive | Small 10-contract quotes, bounded inventory, and a required edge for FOK fills. |
| `conservative_hybrid_market_maker.py` | Conservative hybrid | Small quotes plus reduced two-contract sizing close to expiry for uncertain contracts. |

All models are market makers: they continuously post a bid to buy and an offer to sell around an internally estimated fair value.  They are not buy-and-hold or directional momentum strategies.

## Contract definition and payoff

An option is defined by a set of weighted underlying legs, a strike \(K\), and \(T\) remaining discrete trading steps.  Its observable at expiry is

\[
Y_T = \sum_{j=1}^{m} w_j X_{j,T}.
\]

The payout is the indicator function

\[
V_T = \mathbf{1}\{Y_T \ge K\} =
\begin{cases}
1, & \sum_j w_j X_{j,T} \ge K,\\
0, & \text{otherwise.}
\end{cases}
\]

With no discounting in the supplied game, the model fair price is simply the risk-model probability of paying one:

\[
F_0 = \mathbb{E}[V_T] = \Pr(Y_T \ge K), \qquad 0 \le F_0 \le 1.
\]

`BinaryOption.expiry_valuation` implements this exact terminal payoff.  An `OptionLeg` identifies an underlying and its coefficient \(w_j\).  The available underlyings are AJR (Ajarai), THR (Theriodic), and FED (the rate).

## Statistical model

### Historical calibration

`warm_up` estimates daily company log returns and rate changes from `MarketHistory`:

\[
r_{i,t}=\log(X_{i,t}/X_{i,t-1}), \qquad \Delta R_t=R_t-R_{t-1}.
\]

For each company \(i\), it fits the regularized linear regression

\[
r_{i,t}=\alpha_i+\beta_{i,R}\Delta R_t+\varepsilon_{i,t}.
\]

The implementation obtains the slope from sample covariance and variance (with ridge inflation in the risk-aware versions):

\[
\beta_{i,R}=\frac{\operatorname{Cov}(\Delta R,r_i)}{\operatorname{Var}(\Delta R)(1+\lambda)},
\qquad
\alpha_i=\overline{r_i}-\beta_{i,R}\overline{\Delta R}.
\]

It then decomposes residual covariance into one common sector factor and idiosyncratic variance.  It chooses the company with larger residual variance as the sector-scale anchor, assigns that company sector beta \(1\), and sets

\[
\sigma_S^2=\max\left(\operatorname{Var}(\varepsilon_A),\operatorname{Var}(\varepsilon_T)\right),
\qquad
\beta_{\mathrm{other},S}=\frac{\operatorname{Cov}(\varepsilon_A,\varepsilon_T)}{\sigma_S^2}.
\]

The anchor has zero idiosyncratic variance in this factorization and the other company receives the residual:

\[
\sigma_{\mathrm{other},\mathrm{idio}}^2=
\max\left(\operatorname{Var}(\varepsilon_{\mathrm{other}})-
\beta_{\mathrm{other},S}^2\sigma_S^2,0\right).
\]

The risk-aware files use shrinkage weights so a short history is pulled toward conservative values rather than overfitting noisy estimates.

### Company-price dynamics

Over one step, the simulated / modeled company log return is

\[
\log(X_{i,t+1}/X_{i,t})=
\alpha_i+\beta_{i,R}\Delta R_t+\beta_{i,S}S_t+\epsilon_{i,t},
\]

where \(S_t\sim\mathcal{N}(0,\sigma_S^2)\) is the shared sector shock and \(\epsilon_{i,t}\sim\mathcal{N}(0,\sigma_{i,\mathrm{idio}}^2)\) is idiosyncratic noise.  Thus, conditional on a terminal rate \(r\), each company is treated as lognormal:

\[
\log X_{i,T}\mid R_T=r \sim \mathcal{N}(\mu_i,v_i),
\]

\[
\mu_i=\log X_{i,0}+T\alpha_i+\beta_{i,R}(r-R_0),
\qquad
v_i=T\left(\beta_{i,S}^2\sigma_S^2+\sigma_{i,\mathrm{idio}}^2\right).
\]

The conditional level moments used for multi-leg approximations are

\[
\mathbb{E}[X_{i,T}]=e^{\mu_i+v_i/2},
\qquad
\operatorname{Var}(X_{i,T})=\mathbb{E}[X_{i,T}]^2(e^{v_i}-1).
\]

For two company legs, the shared factor produces the level covariance

\[
\operatorname{Cov}(X_{A,T},X_{T,T})=
\mathbb{E}[X_{A,T}]\mathbb{E}[X_{T,T}]
\left(e^{T\beta_{A,S}\beta_{T,S}\sigma_S^2}-1\right).
\]

### Discrete mean-reverting rate model

FED moves on a non-negative grid with step \(h=0.25\).  At rate \(r\), mean reversion toward target \(r^*\) tilts the move probabilities:

\[
\tau(r)=\kappa(r^*-r),
\quad p_\uparrow(r)=\operatorname{clip}(p_\uparrow+\tau(r),0,1),
\]

\[
p_\downarrow(r)=\operatorname{clip}(p_\downarrow-\tau(r),0,1-p_\uparrow(r)),
\quad p_0(r)=1-p_\uparrow(r)-p_\downarrow(r).
\]

The rate distribution is propagated exactly over all reachable grid states.  If \(\pi_t(r)\) is the probability of rate \(r\) at step \(t\), then

\[
\pi_{t+1}(r+h) {+}= \pi_t(r)p_\uparrow(r),\quad
\pi_{t+1}(\max(r-h,0)) {+}= \pi_t(r)p_\downarrow(r),\quad
\pi_{t+1}(r) {+}= \pi_t(r)p_0(r).
\]

### Binary-option price

For each terminal rate grid state \(r\), the model evaluates the conditional exercise probability \(q(r)\).  The final price is the exact finite mixture

\[
F_0=\sum_r \Pr(R_T=r)q(r).
\]

For a single positive company leg, after moving the rate term to the strike, the conditional probability is the lognormal CDF:

\[
q(r)=\Phi\left(\frac{\mu_i-\log(K'/w_i)}{\sqrt{v_i}}\right),
\]

where \(K'=K-w_Rr\).  For a relative-value pair with one positive and one negative company leg and zero adjusted threshold, it prices the log-ratio exactly under the normal approximation:

\[
q(r)=\Phi\left(\frac{(\mu_A-\mu_T)-\log(|w_T|/w_A)}
{\sqrt{v_A+v_T-2\operatorname{Cov}(\log X_A,\log X_T)}}\right).
\]

Other two-leg combinations use moment matching for \(Y_T\):

\[
m_Y=\sum_i w_i\mathbb{E}[X_i],
\qquad
s_Y^2=\sum_iw_i^2\operatorname{Var}(X_i)+2w_Aw_T\operatorname{Cov}(X_A,X_T),
\]

\[
q(r)\approx\Phi\left(\frac{m_Y-K'}{s_Y}\right).
\]

Here \(\Phi\) is the standard-normal CDF.

## Quoting and inventory control

Each strategy calculates an inventory-skewed reservation price \(R\) and half-spread \(H\), then posts penny-rounded quotes:

\[
\text{bid}=\lfloor100(R-H)\rfloor/100,
\qquad
\text{offer}=\lceil100(R+H)\rceil/100.
\]

For the risk-aware strategies, position risk is measured at the maximum loss of the held side:

\[
\mathrm{risk}(n)=
\begin{cases}
nF_0, & n\ge0,\\
(-n)(1-F_0), & n<0.
\end{cases}
\]

With contract budget \(B\), utilization is \(u=\operatorname{clip}(\mathrm{risk}/B,0,1)\).  The reservation price is shifted away from existing inventory,

\[
R=\operatorname{clip}\left(F_0-0.05\,\operatorname{sign}(n)\,u\,
\max(0.35,4F_0(1-F_0)),0.005,0.995\right),
\]

and the half-spread widens for event uncertainty, maturity, and inventory use:

\[
H=\operatorname{clip}\left(0.012+0.018\sqrt{F_0(1-F_0)}+0.002\sqrt{T}
+\frac{0.025\sqrt{F_0(1-F_0)}}{\sqrt{T}}+0.025u,0.015,0.12\right).
\]

The simpler models use the same general reservation/spread structure, with inventory utilization based on the absolute position limit.  The conservative-hybrid version adds the near-expiry two-lot cap when \(T\le2\) and \(0.35\le R\le0.65\), where binary-outcome uncertainty is highest.

## Capital and fills

Buying one contract has maximum loss equal to its price; selling one has maximum loss equal to \(1-\text{price}\).  The aggressive and hybrid models calculate the available trade budget as

\[
B_{\mathrm{trade}}=\min\left(fC,\max(0,C-rC_0),B_{\mathrm{underlying}}\right),
\]

where \(C\) is current cash, \(C_0\) initial cash, \(r\) the reserve fraction, \(f\) the per-trade fraction, and \(B_{\mathrm{underlying}}\) is remaining collateral headroom for the option’s underlying(s).  They also track long and short collateral separately and release it at expiry.

FOK orders are accepted only when the resulting position remains within its cap, the required collateral fits the applicable capital budget, and the execution has positive modeled edge.  In the risk-aware versions,

\[
\mathrm{edge}=\begin{cases}
\text{order price}-R, & \text{customer buys from the maker},\\
R-\text{order price}, & \text{customer sells to the maker},
\end{cases}
\qquad
\mathrm{return\ on\ risk}=\frac{\mathrm{edge}}{\max(\mathrm{loss\ per\ contract},0.01)}.
\]

They require at least 1¢ edge and 15% return on risk for inventory-increasing fills, relaxing to 0.5¢ and 5% when a fill reduces existing inventory.  The passive variants use an edge buffer of `max(0.01, 0.65 × half_spread)`.

## Risk settings at a glance

| Model | Quote-size cap | Position / capital controls |
| --- | ---: | --- |
| Aggressive | 100 | 35% per-trade fraction, 20% cash reserve, 50% initial-cash underlying cap, 500 emergency position cap. |
| Hybrid | 100 | 25% per-trade fraction, 25% reserve, 45% underlying cap, 500 emergency position cap. |
| Passive | 10 | Position cap is `max(20, min(200, cash / 20))`; cash-only capacity check. |
| Conservative hybrid | 10 (2 near uncertain expiry) | Position cap is `max(1, min(200, 0.15 × cash))`; cash-only capacity check. |

## Typical integration sequence

1. Construct `MarketMaker` with initial underlyings, option list, and cash.
2. Call `warm_up(market_history)` to estimate the statistical parameters.
3. For every option, call `quote(option, counterparty_id)` to receive bid/offer prices and quantities.
4. Call `respond_to_fok(option, fok_order)` to decide whether to trade against a FOK request.
5. If executed, call `on_trade(...)`; after a market step, call `on_step_advance(...)` to update state and settle expired positions.

The files depend only on Python’s standard library (`math`, `random`, `dataclasses`, `enum`, `collections`, and `typing`).
