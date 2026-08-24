import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final

AJARAI_NAME: Final[str] = "AJR"
AJARAI_UNDERLYING_ID: Final[int] = 2
FED_FUNDS_RATE_NAME: Final[str] = "FED"
FED_FUNDS_RATE_UNDERLYING_ID: Final[int] = 1
RATE_STRIKE_GRID: Final[float] = 0.25
THERIODIC_NAME: Final[str] = "THR"
THERIODIC_UNDERLYING_ID: Final[int] = 3

UNDERLYING_NAME_BY_ID: Final[dict[int, str]] = {
    AJARAI_UNDERLYING_ID: AJARAI_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID: FED_FUNDS_RATE_NAME,
    THERIODIC_UNDERLYING_ID: THERIODIC_NAME,
}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class BinaryOption:
    legs: "tuple[OptionLeg, ...]"
    option_id: int
    steps_until_expiry: int
    strike: float

    def __post_init__(self) -> None:
        if self.steps_until_expiry < 0:
            raise ValueError("Steps until expiry must be non-negative")

        if not self.legs:
            raise ValueError("Binary option must have at least one leg")

        underlying_ids: list[int] = [leg.underlying_id for leg in self.legs]
        if len(underlying_ids) != len(set(underlying_ids)):
            raise ValueError("Binary option legs must reference distinct underlyings")

        if any(leg.weight == 0 for leg in self.legs):
            raise ValueError("Binary option leg weights must be non-zero")

    def __str__(self) -> str:
        terms: list[str] = []
        for index, leg in enumerate(self.legs):
            name: str = UNDERLYING_NAME_BY_ID.get(leg.underlying_id, str(leg.underlying_id))
            magnitude: float = abs(leg.weight)
            magnitude_str: str = "" if magnitude == 1 else f"{magnitude:.2f}*"
            if index == 0:
                sign: str = "-" if leg.weight < 0 else ""
            else:
                sign = " - " if leg.weight < 0 else " + "
            terms.append(f"{sign}{magnitude_str}{name}")
        observable_expression: str = "".join(terms)
        return f"{self.option_id} ({self.steps_until_expiry}d {observable_expression} >= {self.strike:.2f})"

    def advance_step(self) -> "BinaryOption":
        if self.steps_until_expiry == 0:
            return self

        return replace(self, steps_until_expiry=self.steps_until_expiry - 1)

    def contract_matches(self, other: "BinaryOption") -> bool:
        return replace(other, option_id=self.option_id) == self

    def expiry_valuation(self, value_by_underlying_id: dict[int, float]) -> float:
        return 1.0 if self.observable_value(value_by_underlying_id) >= self.strike else 0.0

    def observable_value(self, value_by_underlying_id: dict[int, float]) -> float:
        return sum(leg.weight * value_by_underlying_id[leg.underlying_id] for leg in self.legs)


@dataclass(frozen=True)
class FokOrder:
    counterparty_id: int
    option_id: int
    order_type: "OrderType"
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("FOK order price must be non-negative")

        if self.quantity <= 0:
            raise ValueError("FOK order quantity must be positive")


@dataclass(frozen=True)
class MarketHistory:
    values_by_underlying_id: dict[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        lengths: set[int] = {len(values) for values in self.values_by_underlying_id.values()}
        if len(lengths) > 1:
            raise ValueError("All underlyings must have the same number of historical days")

        if lengths and next(iter(lengths)) <= 0:
            raise ValueError("Market history must contain at least one day")

    @property
    def num_days(self) -> int:
        if not self.values_by_underlying_id:
            return 0
        return len(next(iter(self.values_by_underlying_id.values())))


@dataclass(frozen=True)
class MarketParameters:
    ajarai_drift: float
    ajarai_idio_std_dev: float
    ajarai_rate_beta: float
    ajarai_sector_beta: float
    rate_down_probability: float
    rate_reversion_strength: float
    rate_up_probability: float
    sector_std_dev: float
    theriodic_drift: float
    theriodic_idio_std_dev: float
    theriodic_rate_beta: float
    theriodic_sector_beta: float

    rate_step: float = 0.25
    rate_target: float = 2.0

    def __post_init__(self) -> None:
        if self.rate_step <= 0:
            raise ValueError("Rate step must be positive")

        if self.rate_up_probability <= 0 or self.rate_down_probability <= 0:
            raise ValueError("Rate up/down probabilities must both be positive")

        if self.rate_up_probability + self.rate_down_probability > 1:
            raise ValueError("Rate up/down probabilities must not sum to more than 1")

        if self.rate_target < 0:
            raise ValueError("Rate target must be non-negative")

        if not (0 <= self.rate_reversion_strength <= 1):
            raise ValueError("Rate reversion strength must be between 0 and 1")

        if self.ajarai_idio_std_dev < 0 or self.theriodic_idio_std_dev < 0 or self.sector_std_dev < 0:
            raise ValueError("Standard deviations must be non-negative")

    def advance_company_value(
        self,
        current_value: float,
        rate_change: float,
        sector_shock: float,
        *,
        drift: float,
        rate_beta: float,
        sector_beta: float,
        idio_std_dev: float,
    ) -> float:
        idiosyncratic_shock: float = random.gauss(mu=0.0, sigma=idio_std_dev)
        log_return: float = drift + (rate_beta * rate_change) + (sector_beta * sector_shock) + idiosyncratic_shock
        return round(current_value * math.exp(log_return), 2)

    def advance_rate(self, rate_value: float) -> float:
        up_probability, down_probability = self.tilted_rate_probabilities(rate_value)
        draw: float = random.random()
        if draw < up_probability:
            return self.next_rate_value(rate_value, 1)

        if draw < up_probability + down_probability:
            return self.next_rate_value(rate_value, -1)

        return rate_value

    def advance_step(self, value_by_underlying_id: dict[int, float]) -> dict[int, float]:
        current_rate_value: float = value_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
        rate_value: float = self.advance_rate(current_rate_value)
        rate_change: float = round(rate_value - current_rate_value, 2)
        sector_shock: float = random.gauss(mu=0.0, sigma=self.sector_std_dev)
        return {
            FED_FUNDS_RATE_UNDERLYING_ID: rate_value,
            AJARAI_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[AJARAI_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.ajarai_drift,
                rate_beta=self.ajarai_rate_beta,
                sector_beta=self.ajarai_sector_beta,
                idio_std_dev=self.ajarai_idio_std_dev,
            ),
            THERIODIC_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[THERIODIC_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.theriodic_drift,
                rate_beta=self.theriodic_rate_beta,
                sector_beta=self.theriodic_sector_beta,
                idio_std_dev=self.theriodic_idio_std_dev,
            ),
        }

    def next_rate_value(self, rate_value: float, num_grid_steps: int) -> float:
        return max(round(rate_value + num_grid_steps * self.rate_step, 2), 0.0)

    def tilted_rate_probabilities(self, rate_value: float) -> tuple[float, float]:
        tilt: float = self.rate_reversion_strength * (self.rate_target - rate_value)
        up_probability: float = min(max(self.rate_up_probability + tilt, 0.0), 1.0)
        down_probability: float = min(max(self.rate_down_probability - tilt, 0.0), 1.0 - up_probability)
        return up_probability, down_probability


@dataclass(frozen=True)
class OptionLeg:
    underlying_id: int
    weight: float


class OrderType(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Position:
    def __init__(self) -> None:
        self.option_quantity_by_option_id: dict[int, int] = defaultdict(int)

    def add_option_quantity(self, option_id: int, quantity: int) -> None:
        self.option_quantity_by_option_id[option_id] += quantity


@dataclass(frozen=True)
class Quote:
    bid_price: float
    bid_quantity: int
    offer_price: float
    offer_quantity: int

    def __post_init__(self) -> None:
        if self.bid_quantity <= 0 or self.offer_quantity <= 0:
            raise ValueError("Quote quantities must be positive")

        if not (0.0 <= self.bid_price <= 1.0 and 0.0 <= self.offer_price <= 1.0):
            raise ValueError("Quote prices must be between 0 and 1")

        if self.bid_price >= self.offer_price:
            raise ValueError("Quote bid price must be less than offer price")

        if any(abs(round(price * 100) - price * 100) > 1e-6 for price in (self.bid_price, self.offer_price)):
            raise ValueError("Quote prices must be in whole pennies (multiples of 0.01)")


@dataclass(frozen=True)
class Underlying:
    name: str
    underlying_id: int
    value: float

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Underlying):
            return False
        return self.underlying_id == other.underlying_id

# ============================================================================
# YOUR MARKET MAKER -- fill in the six stubbed methods below
# ============================================================================


class MarketMaker:
    def __init__(
        self,
        underlying_initial_state: list[Underlying],
        option_initial_state: list[BinaryOption],
        cash_balance: float,
    ) -> None:
        self.underlying_state: list[Underlying] = underlying_initial_state
        self.active_option_state: list[BinaryOption] = option_initial_state
        self.cash_balance: float = cash_balance
        self.initial_cash: float = cash_balance
        self.position: Position = Position()
        self.params = None
        self.max_position = max(1, min(200, int(0.15 * max(cash_balance, 1.0))))
        self._rate_dist_cache = {}

    def on_step_advance(
        self, new_underlying_state: list[Underlying], new_option_state: list[BinaryOption]
    ) -> None:
        new_by_id = {option.option_id: option for option in new_option_state}
        terminal_values = {u.underlying_id: u.value for u in new_underlying_state}
        for option in self.active_option_state:
            expired = option.option_id not in new_by_id or (
                option.steps_until_expiry > 0 and new_by_id[option.option_id].steps_until_expiry == 0
            )
            if expired:
                quantity = self.position.option_quantity_by_option_id[option.option_id]
                settlement = option.expiry_valuation(terminal_values)
                self.cash_balance += quantity * settlement if quantity > 0 else (-quantity) * (1.0 - settlement)
                self.position.option_quantity_by_option_id[option.option_id] = 0
        self.underlying_state = new_underlying_state
        self.active_option_state = new_option_state

    def on_trade(self, option: BinaryOption, price: float, quantity: int, counterparty_id: int) -> None:
        self.position.add_option_quantity(option.option_id, quantity)
        if quantity > 0:
            self.cash_balance -= quantity * price
        else:
            self.cash_balance -= (-quantity) * (1.0 - price)

    @property
    def name(self) -> str:
        return "Aritra-ExactMixture-MM"

    @staticmethod
    def _clip(x, lo, hi):
        return min(max(x, lo), hi)

    @staticmethod
    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    @classmethod
    def _variance(cls, xs):
        if len(xs) < 2:
            return 0.0
        m = cls._mean(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

    @classmethod
    def _covariance(cls, xs, ys):
        if len(xs) < 2:
            return 0.0
        mx, my = cls._mean(xs), cls._mean(ys)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)

    @classmethod
    def _regression(cls, ys, xs, ridge_fraction=0.0):
        vx = cls._variance(xs)
        denominator = vx * (1.0 + max(ridge_fraction, 0.0))
        beta = cls._covariance(xs, ys) / denominator if denominator > 1e-14 else 0.0
        return cls._mean(ys) - beta * cls._mean(xs), beta

    def warm_up(self, market_history: MarketHistory) -> None:
        values = market_history.values_by_underlying_id
        rates = list(values[FED_FUNDS_RATE_UNDERLYING_ID])
        ajr = list(values[AJARAI_UNDERLYING_ID])
        thr = list(values[THERIODIC_UNDERLYING_ID])

        rate_changes = [rates[i] - rates[i - 1] for i in range(1, len(rates))]
        rate_states = rates[:-1]
        target = 2.0

        if rate_changes:
            up = [1.0 if x > 1e-9 else 0.0 for x in rate_changes]
            down = [1.0 if x < -1e-9 else 0.0 for x in rate_changes]
            tilts = [target - x for x in rate_states]
            up0, up_slope = self._regression(up, tilts, ridge_fraction=0.10)
            down0, down_slope = self._regression(down, tilts, ridge_fraction=0.10)
            sample_weight = len(rate_changes) / (len(rate_changes) + 75.0)
            raw_strength = self._clip((up_slope - down_slope) / 2.0, 0.0, 1.0)
            strength = sample_weight * raw_strength
            empirical_up = (sum(up) + 2.0) / (len(up) + 6.0)
            empirical_down = (sum(down) + 2.0) / (len(down) + 6.0)
            p_up = self._clip(sample_weight * up0 + (1.0 - sample_weight) * empirical_up, 0.001, 0.998)
            p_down_raw = sample_weight * down0 + (1.0 - sample_weight) * empirical_down
            p_down = self._clip(p_down_raw, 0.001, 0.999 - p_up)
        else:
            p_up, p_down, strength = 0.25, 0.25, 0.0

        def log_returns(series):
            return [math.log(series[i] / series[i - 1]) for i in range(1, len(series))]

        ajr_ret, thr_ret = log_returns(ajr), log_returns(thr)
        ajr_drift, ajr_rate_beta = self._regression(ajr_ret, rate_changes, ridge_fraction=0.05)
        thr_drift, thr_rate_beta = self._regression(thr_ret, rate_changes, ridge_fraction=0.05)
        return_weight = len(rate_changes) / (len(rate_changes) + 50.0) if rate_changes else 0.0
        ajr_drift *= return_weight
        thr_drift *= return_weight
        ajr_resid = [y - ajr_drift - ajr_rate_beta * x for y, x in zip(ajr_ret, rate_changes)]
        thr_resid = [y - thr_drift - thr_rate_beta * x for y, x in zip(thr_ret, rate_changes)]

        va, vt = self._variance(ajr_resid), self._variance(thr_resid)
        covariance_weight = len(ajr_resid) / (len(ajr_resid) + 30.0) if ajr_resid else 0.0
        cov = self._covariance(ajr_resid, thr_resid) * covariance_weight
        sector_sd = 1.0
        if va >= vt and va > 1e-14:
            ajr_sector_beta = math.sqrt(va)
            thr_sector_beta = cov / ajr_sector_beta
            ajr_idio_var = 0.0
            thr_idio_var = max(vt - thr_sector_beta ** 2, 0.0)
        elif vt > 1e-14:
            thr_sector_beta = math.sqrt(vt)
            ajr_sector_beta = cov / thr_sector_beta
            thr_idio_var = 0.0
            ajr_idio_var = max(va - ajr_sector_beta ** 2, 0.0)
        else:
            ajr_sector_beta = thr_sector_beta = 0.0
            ajr_idio_var = thr_idio_var = 0.0

        self.params = MarketParameters(
            ajarai_drift=ajr_drift,
            ajarai_idio_std_dev=math.sqrt(max(ajr_idio_var, 0.0)),
            ajarai_rate_beta=ajr_rate_beta,
            ajarai_sector_beta=ajr_sector_beta,
            rate_down_probability=p_down,
            rate_reversion_strength=strength,
            rate_up_probability=p_up,
            sector_std_dev=sector_sd,
            theriodic_drift=thr_drift,
            theriodic_idio_std_dev=math.sqrt(max(thr_idio_var, 0.0)),
            theriodic_rate_beta=thr_rate_beta,
            theriodic_sector_beta=thr_sector_beta,
        )

    @staticmethod
    def _normal_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _rate_distribution(self, parameters, start, steps):
        cache_key = (parameters, round(start, 2), steps)
        cached = self._rate_dist_cache.get(cache_key)
        if cached is not None:
            return cached
        dist = {round(start, 2): 1.0}
        for _ in range(steps):
            nxt = defaultdict(float)
            for rate, probability in dist.items():
                pu, pd = parameters.tilted_rate_probabilities(rate)
                nxt[parameters.next_rate_value(rate, 1)] += probability * pu
                nxt[parameters.next_rate_value(rate, -1)] += probability * pd
                nxt[rate] += probability * max(0.0, 1.0 - pu - pd)
            dist = dict(nxt)
        self._rate_dist_cache[cache_key] = dist
        return dist

    def _conditional_probability(self, parameters, option, terminal_rate, current):
        steps = option.steps_until_expiry
        rate0 = current[FED_FUNDS_RATE_UNDERLYING_ID]
        constant = 0.0
        company_legs = []
        for leg in option.legs:
            if leg.underlying_id == FED_FUNDS_RATE_UNDERLYING_ID:
                constant += leg.weight * terminal_rate
            else:
                company_legs.append(leg)

        threshold = option.strike - constant
        if not company_legs:
            return 1.0 if constant >= option.strike else 0.0

        moments = {}
        log_cov = 0.0
        for leg in company_legs:
            if leg.underlying_id == AJARAI_UNDERLYING_ID:
                drift = parameters.ajarai_drift
                rb = parameters.ajarai_rate_beta
                sb = parameters.ajarai_sector_beta
                iv = parameters.ajarai_idio_std_dev ** 2
            else:
                drift = parameters.theriodic_drift
                rb = parameters.theriodic_rate_beta
                sb = parameters.theriodic_sector_beta
                iv = parameters.theriodic_idio_std_dev ** 2
            lv = steps * (sb * sb * parameters.sector_std_dev ** 2 + iv)
            lm = math.log(max(current[leg.underlying_id], 1e-12)) + steps * drift + rb * (terminal_rate - rate0)
            mean = math.exp(lm + 0.5 * lv)
            variance = mean * mean * max(math.exp(lv) - 1.0, 0.0)
            moments[leg.underlying_id] = (mean, variance, sb, lm, lv)

        if len(company_legs) == 1:
            leg = company_legs[0]
            _, _, _, lm, lv = moments[leg.underlying_id]
            cutoff = threshold / leg.weight
            if leg.weight > 0:
                if cutoff <= 0:
                    return 1.0
                if lv <= 1e-14:
                    return 1.0 if math.exp(lm) >= cutoff else 0.0
                return self._normal_cdf((lm - math.log(cutoff)) / math.sqrt(lv))
            if cutoff <= 0:
                return 0.0
            if lv <= 1e-14:
                return 1.0 if math.exp(lm) <= cutoff else 0.0
            return self._normal_cdf((math.log(cutoff) - lm) / math.sqrt(lv))

        if len(company_legs) == 2 and abs(threshold) <= 1e-12:
            positive = next((leg for leg in company_legs if leg.weight > 0), None)
            negative = next((leg for leg in company_legs if leg.weight < 0), None)
            if positive is not None and negative is not None:
                _, _, sbp, lmp, lvp = moments[positive.underlying_id]
                _, _, sbn, lmn, lvn = moments[negative.underlying_id]
                covariance = steps * sbp * sbn * parameters.sector_std_dev ** 2
                ratio_variance = max(lvp + lvn - 2.0 * covariance, 0.0)
                cutoff_log = math.log(abs(negative.weight) / positive.weight)
                ratio_mean = lmp - lmn
                if ratio_variance <= 1e-14:
                    return 1.0 if ratio_mean >= cutoff_log else 0.0
                return self._normal_cdf((ratio_mean - cutoff_log) / math.sqrt(ratio_variance))

        mean_sum = sum(leg.weight * moments[leg.underlying_id][0] for leg in company_legs)
        var_sum = sum(leg.weight ** 2 * moments[leg.underlying_id][1] for leg in company_legs)
        if len(company_legs) == 2:
            a, b = company_legs
            ma, _, sba, _, _ = moments[a.underlying_id]
            mb, _, sbb, _, _ = moments[b.underlying_id]
            log_cov = steps * sba * sbb * parameters.sector_std_dev ** 2
            covariance = ma * mb * (math.exp(log_cov) - 1.0)
            var_sum += 2.0 * a.weight * b.weight * covariance

        sd_sum = math.sqrt(max(var_sum, 1e-14))
        return self._normal_cdf((mean_sum - threshold) / sd_sum)

    def price_option_from_parameters(
        self, market_parameters: MarketParameters, option: BinaryOption
    ) -> float:
        current = {u.underlying_id: u.value for u in self.underlying_state}
        if option.steps_until_expiry == 0:
            return option.expiry_valuation(current)
        rate0 = current[FED_FUNDS_RATE_UNDERLYING_ID]
        dist = self._rate_distribution(market_parameters, rate0, option.steps_until_expiry)
        value = sum(
            probability * self._conditional_probability(market_parameters, option, rate, current)
            for rate, probability in dist.items()
        )
        return self._clip(value, 0.0, 1.0)

    def price_option(self, option: BinaryOption) -> float:
        if self.params is None:
            return 0.5
        return self.price_option_from_parameters(self.params, option)

    def _reservation_and_half_spread(self, option):
        fair = self.price_option(option)
        position = self.position.option_quantity_by_option_id[option.option_id]
        utilization = min(abs(position) / max(self.max_position, 1), 1.0)
        skew = 0.0015 * position * max(0.35, 4.0 * fair * (1.0 - fair))
        reservation = self._clip(fair - skew, 0.005, 0.995)
        event_risk = math.sqrt(max(fair * (1.0 - fair), 0.0))
        time_factor = math.sqrt(max(option.steps_until_expiry, 1))
        expiry_risk = 0.025 * event_risk / time_factor
        half = self._clip(
            0.012 + 0.018 * event_risk + 0.002 * time_factor + expiry_risk + 0.025 * utilization,
            0.015,
            0.12,
        )
        return reservation, half

    def quote(self, option: BinaryOption, counterparty_id: int) -> Quote:
        reservation, half = self._reservation_and_half_spread(option)
        bid = math.floor(max(0.0, reservation - half) * 100.0 + 1e-9) / 100.0
        offer = math.ceil(min(1.0, reservation + half) * 100.0 - 1e-9) / 100.0
        bid = self._clip(bid, 0.0, 0.99)
        offer = self._clip(offer, 0.01, 1.0)
        if bid >= offer:
            bid, offer = max(0.0, bid - 0.01), min(1.0, offer + 0.01)
        position = self.position.option_quantity_by_option_id[option.option_id]
        if position >= self.max_position:
            bid = 0.0
        if position <= -self.max_position:
            offer = 1.0
        bid_capacity = int(self.cash_balance / bid) if bid > 0 else self.max_position
        offer_risk = 1.0 - offer
        offer_capacity = int(self.cash_balance / offer_risk) if offer_risk > 0 else self.max_position
        bid_qty = min(10, max(0, self.max_position - position), bid_capacity)
        offer_qty = min(10, max(0, self.max_position + position), offer_capacity)
        if option.steps_until_expiry <= 2 and 0.35 <= reservation <= 0.65:
            bid_qty = min(bid_qty, 2)
            offer_qty = min(offer_qty, 2)
        if bid_qty < 1:
            bid, bid_qty = 0.0, 1
        if offer_qty < 1:
            offer, offer_qty = 1.0, 1
        return Quote(bid, max(1, bid_qty), offer, max(1, offer_qty))

    def respond_to_fok(self, option: BinaryOption, fok_order: FokOrder) -> bool:
        reservation, half = self._reservation_and_half_spread(option)
        position = self.position.option_quantity_by_option_id[option.option_id]
        edge_buffer = max(0.01, half * 0.65)
        if fok_order.order_type == OrderType.BUY:
            projected = position - fok_order.quantity
            capital_needed = fok_order.quantity * (1.0 - fok_order.price)
            return (
                projected >= -self.max_position
                and capital_needed <= self.cash_balance
                and fok_order.price >= reservation + edge_buffer
            )
        projected = position + fok_order.quantity
        capital_needed = fok_order.quantity * fok_order.price
        return (
            projected <= self.max_position
            and capital_needed <= self.cash_balance
            and fok_order.price <= reservation - edge_buffer
        )

