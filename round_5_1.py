from typing import Any, List, TypeAlias
import json
import jsonpickle
import numpy as np
import math
from statistics import NormalDist
from abc import abstractmethod
from enum import IntEnum

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

def brentq(f, a, b, tol=1e-10, max_iter=100):
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError("Function must have different signs at endpoints a and b")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = e = b - a

    for iteration in range(max_iter):
        if fb * fc > 0:
            c = a
            fc = fa
            d = e = b - a

        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol1 = 2 * tol * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)

        if abs(m) <= tol1 or fb == 0:
            return b

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                # Secant method
                p = 2 * m * s
                q = 1 - s
            else:
                # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)

            if p > 0:
                q = -q
            p = abs(p)

            if 2 * p < min(3 * m * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = e = m
        else:
            d = e = m

        a = b
        fa = fb

        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if m > 0 else -tol1

        fb = f(b)

    raise RuntimeError("Maximum number of iterations exceeded in brentq")

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = "KELP"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

BASKET1_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}

BASKET2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}

# TODO: DAY = 5
DAY = 5

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,        # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,             # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 40,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "window_size": 50,
        "deviation_threshold": 0.01,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "window_size": 20,
        "deviation_threshold": 0.035,
    },
    Product.DJEMBES: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 2,
    },
    Product.CROISSANTS: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "synthetic_weight": 0.000001,
    },
    Product.JAMS: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "synthetic_weight": 0,
    },
    Product.PICNIC_BASKET1: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 2,
        "synthetic_weight": 0.04,
        "volatility_window_size": 10,
        "adverse_volatility": 0.1,
    },
    Product.PICNIC_BASKET2: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 2,
        "synthetic_weight": 0.03,
        "volatility_window_size": 10,
        "adverse_volatility": 0.1,
    },
    Product.VOLCANIC_ROCK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 5,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 5,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 5,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 5,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 5,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 2,
        "make_min_edge": 1,
        "make_probability": 0.6,
        "init_make_edge": 2,
        "min_edge": 0.5,
        "volume_avg_timestamp": 5,
        "volume_bar": 75,
        "dec_edge_discount": 0.8,
        "step_size": 0.5,
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 2,
        "window_size": 50,
        "deviation_threshold": 0.05,
        "model_weight": 0.02,
    },
}

VOLCANIC_ROCK_VOUCHER_STRIKE = {
    Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
    Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
    Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
    Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
    Product.VOLCANIC_ROCK_VOUCHER_10500: 10500,
}

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState, traderObject) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state, traderObject)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2
    
    def filtered_mid(
        self,
        product: str,
        order_depth: OrderDepth,
        adverse_volume: int
    ) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_asks = [
            price for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price]) >= adverse_volume
        ]
        filtered_bids = [
            price for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price]) >= adverse_volume
        ]
        best_filtered_ask = min(filtered_asks) if filtered_asks else None
        best_filtered_bid = max(filtered_bids) if filtered_bids else None

        if best_filtered_ask is not None and best_filtered_bid is not None:
            return (best_filtered_ask + best_filtered_bid) / 2
        return (best_ask + best_bid) / 2

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class Signal(IntEnum):
    NEUTRAL = 0
    SHORT = 1
    LONG = 2

class SignalStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

    @abstractmethod
    def get_signal(self, state: TradingState) -> Signal | None:
        raise NotImplementedError()

    def act(self, state: TradingState, traderObject) -> None:
        traderObject.setdefault(self.symbol, {})
        
        traderObject[self.symbol]["signal"] = self.get_signal(state, traderObject)

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if traderObject[self.symbol]["signal"] == Signal.NEUTRAL:
            if position < 0:
                self.buy(self.get_buy_price(order_depth), -position)
            elif position > 0:
                self.sell(self.get_sell_price(order_depth), position)
        elif traderObject[self.symbol]["signal"] == Signal.SHORT:
            self.sell(self.get_sell_price(order_depth), self.limit + position)
        elif traderObject[self.symbol]["signal"] == Signal.LONG:
            self.buy(self.get_buy_price(order_depth), self.limit - position)

    def get_buy_price(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders.keys())

    def get_sell_price(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders.keys())

class CroissantsStrategy(SignalStrategy):
    def get_signal(self, state: TradingState, traderObject) -> Signal | None:
        mid = self.filtered_mid(self.symbol, state.order_depths[self.symbol], 15)

        for trade in state.market_trades.get(self.symbol, []):
            if trade.timestamp == state.timestamp - 100:
                if trade.buyer == "Olivia":
                    traderObject[self.symbol]["signal"] = Signal.LONG
                    traderObject[self.symbol]["signal_timestamp"] = state.timestamp
                    traderObject[self.symbol]["signal_start_mid"] = mid
                if trade.seller == "Olivia":
                    traderObject[self.symbol]["signal"] = Signal.SHORT
                    traderObject[self.symbol]["signal_timestamp"] = state.timestamp
                    traderObject[self.symbol]["signal_start_mid"] = mid
                
        if (
            "signal" not in traderObject[self.symbol]
            or (
                traderObject[self.symbol]["signal"] != Signal.NEUTRAL
                and traderObject[self.symbol]["signal_timestamp"] < state.timestamp - 50_000
            ) or (
                traderObject[self.symbol]["signal"] == Signal.SHORT
                and traderObject[self.symbol]["signal_start_mid"] < mid * 0.99
            ) or (
                traderObject[self.symbol]["signal"] == Signal.LONG
                and traderObject[self.symbol]["signal_start_mid"] * 0.99 > mid
            )
        ):
            traderObject[self.symbol]["signal"] = Signal.NEUTRAL
            traderObject[self.symbol]["signal_timestamp"] = state.timestamp
            traderObject[self.symbol]["signal_start_mid"] = mid

        return traderObject[self.symbol]["signal"]

class SquidInkStrategy(SignalStrategy):
    def get_signal(self, state: TradingState, traderObject) -> Signal | None:
        mid = self.filtered_mid(self.symbol, state.order_depths[self.symbol], 15)

        for trade in state.market_trades.get(self.symbol, []):
            if trade.timestamp == state.timestamp - 100:
                if trade.buyer == "Olivia":
                    traderObject[self.symbol]["signal"] = Signal.LONG
                    traderObject[self.symbol]["signal_timestamp"] = state.timestamp
                    traderObject[self.symbol]["signal_start_mid"] = mid
                    traderObject[self.symbol]["is_counterparty"] = True
                if trade.seller == "Olivia":
                    traderObject[self.symbol]["signal"] = Signal.SHORT
                    traderObject[self.symbol]["signal_timestamp"] = state.timestamp
                    traderObject[self.symbol]["signal_start_mid"] = mid
                    traderObject[self.symbol]["is_counterparty"] = True
                
        if (
            "signal" not in traderObject[self.symbol]
            or traderObject[self.symbol]["is_counterparty"] == False
            or (
                traderObject[self.symbol]["signal"] != Signal.NEUTRAL
                and traderObject[self.symbol]["signal_timestamp"] < state.timestamp - 50_000
            ) or (
                traderObject[self.symbol]["signal"] == Signal.SHORT
                and traderObject[self.symbol]["signal_start_mid"] < mid * 0.99
            ) or (
                traderObject[self.symbol]["signal"] == Signal.LONG
                and traderObject[self.symbol]["signal_start_mid"] * 0.99 > mid
            )
        ):
            traderObject[self.symbol]["signal"] = Signal.NEUTRAL
            traderObject[self.symbol]["signal_timestamp"] = state.timestamp
            traderObject[self.symbol]["signal_start_mid"] = mid
            traderObject[self.symbol]["is_counterparty"] = False

        window_size = 20
        price_history = traderObject.setdefault("squid_ink_price_history", [])
        price_history.append(mid)
        if len(price_history) > window_size:
            price_history.pop(0)

        if traderObject[self.symbol]["signal"] == Signal.NEUTRAL and len(price_history) == window_size:
            sorted_price_history = sorted(price_history)

            # take middle 50%
            lower_index = window_size // 4
            upper_index = window_size - lower_index
            middle_prices = sorted_price_history[lower_index:upper_index]  
            
            truncated_mean_price = sum(middle_prices) / len(middle_prices)

            deviation = (mid - truncated_mean_price) / truncated_mean_price
            deviation_threshold = 0.04
            if deviation > deviation_threshold:
                print("LONG", state.timestamp, mid, truncated_mean_price)
                traderObject[self.symbol]["signal"] = Signal.LONG
                traderObject[self.symbol]["signal_timestamp"] = state.timestamp
                traderObject[self.symbol]["signal_start_mid"] = mid
                traderObject[self.symbol]["is_counterparty"] = False
            if deviation < -1 * deviation_threshold:
                print("SHORT", state.timestamp, mid, truncated_mean_price)
                traderObject[self.symbol]["signal"] = Signal.SHORT
                traderObject[self.symbol]["signal_timestamp"] = state.timestamp
                traderObject[self.symbol]["signal_start_mid"] = mid
                traderObject[self.symbol]["is_counterparty"] = False

        return traderObject[self.symbol]["signal"]

class MacaronsStrategy(SignalStrategy):
    def get_signal(self, state: TradingState, traderObject) -> Signal | None:
        CSI = 43
        sunlight_window = 20

        macaron_observations = state.observations.conversionObservations[self.symbol]

        history = traderObject.setdefault("macarons_feature_history", [])
        history.append({
            "sugarPrice": macaron_observations.sugarPrice,
            "sunlightIndex": macaron_observations.sunlightIndex,
        })
        if len(history) > sunlight_window:
            history.pop(0)

        if macaron_observations.sunlightIndex < CSI and len(history) >= sunlight_window:
            sunlight_now = history[-1]["sunlightIndex"]
            sunlight_past = history[-sunlight_window]["sunlightIndex"]
            sunlight_delta = sunlight_now - sunlight_past

            if sunlight_delta < 0:
                return Signal.LONG
            if sunlight_delta > 0:
                return Signal.SHORT

        return Signal.NEUTRAL

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50,
            Product.KELP: 50,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.DJEMBES: 60,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            Product.MAGNIFICENT_MACARONS: 75,
        }

        self.CONVERSION_LIMIT = {
            Product.MAGNIFICENT_MACARONS: 10,
        }

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if product == Product.SQUID_INK:
                        quantity = position_limit - position
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if product == Product.SQUID_INK:
                        quantity = position_limit - position
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            traderObject["kelp_last_price"] = mmmid_price

            return mmmid_price
        return None
    
    def filtered_mid(
        self,
        product: str,
        order_depth: OrderDepth
    ) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_asks = [
            price for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price]) >= self.params[product]["adverse_volume"]
        ]
        filtered_bids = [
            price for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price]) >= self.params[product]["adverse_volume"]
        ]
        best_filtered_ask = min(filtered_asks) if filtered_asks else None
        best_filtered_bid = max(filtered_bids) if filtered_bids else None

        if best_filtered_ask is not None and best_filtered_bid is not None:
            return (best_filtered_ask + best_filtered_bid) / 2
        return (best_ask + best_bid) / 2
    
    def basket1_fair_value(
        self,
        order_depth: OrderDepth,
        djembes_order_depth: OrderDepth,
        croissants_order_depth: OrderDepth,
        jams_order_depth: OrderDepth,
        position: int,
        traderObject
    ) -> float:
        mid = self.filtered_mid(Product.PICNIC_BASKET1, order_depth)
        djembes_mid = self.filtered_mid(Product.DJEMBES, djembes_order_depth)
        croissants_mid = self.filtered_mid(Product.CROISSANTS, croissants_order_depth)
        jams_mid = self.filtered_mid(Product.JAMS, jams_order_depth)

        if mid is None:
            return None
        if djembes_mid is None or croissants_mid is None or jams_mid is None:
            return mid
        
        basket1_synthetic_mid = (
            djembes_mid * BASKET1_WEIGHTS[Product.DJEMBES] + 
            croissants_mid * BASKET1_WEIGHTS[Product.CROISSANTS] + 
            jams_mid * BASKET1_WEIGHTS[Product.JAMS]
        )
        
        # code to check spread volatility
        # idea: if the spread between synthetic and real recently changed a lot,
        # don't trade based on the synthetic mid

        # synthetic_spread = abs(basket1_synthetic_mid - mid)
        # if "basket1_synthetic_spread_history" not in traderObject:
        #     traderObject["basket1_synthetic_spread_history"] = []
        # traderObject["basket1_synthetic_spread_history"].append(synthetic_spread)
        # window_size = self.params[Product.PICNIC_BASKET1]["volatility_window_size"]
        # spread_history = traderObject["basket1_synthetic_spread_history"][-window_size:]
        # traderObject["basket1_synthetic_spread_history"] = spread_history

        # if (len(spread_history) < window_size):
        #     return mid
        
        # spread_history = np.array(spread_history, dtype=float)
        # prev = spread_history[:-1]
        # curr = spread_history[1:]
        # with np.errstate(divide='ignore', invalid='ignore'):    # prevent divide-by-zero and log(0)
        #     ratios = np.where(prev != 0, curr / prev, np.nan)
        #     log_returns = np.log(ratios)
        #     clean_log_returns = log_returns[np.isfinite(log_returns)]
        # spread_volatility = np.std(clean_log_returns)

        # if (spread_volatility > self.params[Product.PICNIC_BASKET1]["adverse_volatility"]):
        #     return mid

        # put exponentially less weight on synthetic mid if near position limit
        synthetic_weight = self.params[Product.PICNIC_BASKET1]["synthetic_weight"]
        if abs(position) > 0:
            position_limit_ratio = abs(position) / self.LIMIT[Product.PICNIC_BASKET1]
            decay_factor = math.exp(-1 * position_limit_ratio)
            synthetic_weight *= decay_factor
        return (1 - synthetic_weight) * mid + synthetic_weight * basket1_synthetic_mid
    
    def basket2_fair_value(
        self,
        order_depth: OrderDepth,
        croissants_order_depth: OrderDepth,
        jams_order_depth: OrderDepth,
        position: int,
        traderObject
    ) -> float:
        mid = self.filtered_mid(Product.PICNIC_BASKET1, order_depth)
        croissants_mid = self.filtered_mid(Product.CROISSANTS, croissants_order_depth)
        jams_mid = self.filtered_mid(Product.JAMS, jams_order_depth)

        if mid is None:
            return None
        if croissants_mid is None or jams_mid is None:
            return mid
        
        basket2_synthetic_mid = ( 
            croissants_mid * BASKET2_WEIGHTS[Product.CROISSANTS] +
            jams_mid * BASKET2_WEIGHTS[Product.JAMS]
        )

        # code to check spread volatility
        # idea: if the spread between synthetic and real recently changed a lot,
        # don't trade based on the synthetic mid

        # synthetic_spread = abs(basket2_synthetic_mid - mid)
        # if "basket2_synthetic_spread_history" not in traderObject:
        #     traderObject["basket2_synthetic_spread_history"] = []
        # traderObject["basket2_synthetic_spread_history"].append(synthetic_spread)
        # window_size = self.params[Product.PICNIC_BASKET2]["volatility_window_size"]
        # spread_history = traderObject["basket2_synthetic_spread_history"][-window_size:]
        # traderObject["basket2_synthetic_spread_history"] = spread_history

        # if (len(spread_history) < window_size):
        #     return mid
        
        # spread_history = np.array(spread_history, dtype=float)
        # prev = spread_history[:-1]
        # curr = spread_history[1:]
        # with np.errstate(divide='ignore', invalid='ignore'):    # prevent divide-by-zero and log(0)
        #     ratios = np.where(prev != 0, curr / prev, np.nan)
        #     log_returns = np.log(ratios)
        #     clean_log_returns = log_returns[np.isfinite(log_returns)]
        # spread_volatility = np.std(clean_log_returns)

        # if (spread_volatility > self.params[Product.PICNIC_BASKET2]["adverse_volatility"]):
        #     return mid

        # put exponentially less weight on synthetic mid if near position limit
        synthetic_weight = self.params[Product.PICNIC_BASKET2]["synthetic_weight"]
        if abs(position) > 0:
            position_limit_ratio = abs(position) / self.LIMIT[Product.PICNIC_BASKET2]
            decay_factor = math.exp(-1 * position_limit_ratio)
            synthetic_weight *= decay_factor
        return (1 - synthetic_weight) * mid + synthetic_weight * basket2_synthetic_mid

    def black_scholes(self, S, K, T, r, sigma):
        N = NormalDist().cdf
        d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    
    def implied_volatility(self, call_price, spot, strike, time_to_expiry):
        # define the equation where the root is the implied volatility
        def equation(volatility):
            estimated_price = self.black_scholes(spot, strike, time_to_expiry, 0, volatility)
            return estimated_price - call_price

        # using Brent's method to find the root of the equation
        implied_vol = brentq(equation, 1e-6, 3)
        return implied_vol

    def volcanic_rock_voucher_fair_value(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_order_depths,
        timestamp: int,
        traderObject,
    ) -> dict:
        S = self.filtered_mid(Product.VOLCANIC_ROCK, volcanic_rock_order_depth)
        if S is None:
            return None

        ticks_per_day = 1_000_000
        total_ticks = (8 - DAY) * ticks_per_day
        T = ((total_ticks - timestamp) / total_ticks) * ((8 - DAY) / 365)

        voucher_data = {}
        m_list, v_list = [], []

        for product, K in VOLCANIC_ROCK_VOUCHER_STRIKE.items():
            voucher_mid = self.filtered_mid(product, volcanic_rock_voucher_order_depths[product])
            if voucher_mid is None:
                continue
            try:
                v_t = self.implied_volatility(voucher_mid, S, K, T)
                m_t = np.log(K / S) / np.sqrt(T)
                voucher_data[product] = {"K": K, "mid": voucher_mid, "m_t": m_t, "v_t": v_t}
                m_list.append(float(m_t))
                v_list.append(float(v_t))

            except Exception:
                continue

        if len(m_list) < 3:
            return None

        # rolling smile history
        smile_history = traderObject.setdefault("volcanic_rock_voucher_smile_history", [])
        smile_history.append([m_list, v_list])
        if len(smile_history) > 10:
            smile_history.pop(0)

        # combine all m/v from recent history
        all_m = [m for m_hist, _ in smile_history[-10:] for m in m_hist]
        all_v = [v for _, v_hist in smile_history[-10:] for v in v_hist]
        a, b, c = np.polyfit(all_m, all_v, deg=2)

        zscore_history = traderObject.setdefault("zscore_history", {})
        signals = {}

        for product, data in voucher_data.items():
            fitted_iv = max(a * data["m_t"] ** 2 + b * data["m_t"] + c, 0.01)
            iv_diff = data["v_t"] - fitted_iv

            iv_diff_list = zscore_history.setdefault(product, [])
            iv_diff_list.append(float(iv_diff))

            if len(iv_diff_list) > 10:
                iv_diff_list.pop(0)

            rolling_mean = np.mean(iv_diff_list)
            rolling_std = np.std(iv_diff_list)
            rolling_std = rolling_std if rolling_std > 1e-6 else 1.0
            zscore = (iv_diff - rolling_mean) / rolling_std

            if abs(zscore) > 1.1:
                signals[product] = self.black_scholes(S, data["K"], T, r=0, sigma=fitted_iv)
            else:
                signals[product] = None

        return signals

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        logger = Logger()

        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            rainforest_resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                rainforest_resin_position,
            )
            rainforest_resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
            )
            rainforest_resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_resin_take_orders + rainforest_resin_clear_orders + rainforest_resin_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )

        if Product.PICNIC_BASKET1 in self.params and Product.PICNIC_BASKET1 in state.order_depths:
            basket1_position = (
                state.position[Product.PICNIC_BASKET1]
                if Product.PICNIC_BASKET1 in state.position
                else 0
            )
            basket1_fair_value = self.basket1_fair_value(
                state.order_depths[Product.PICNIC_BASKET1],
                state.order_depths[Product.DJEMBES],
                state.order_depths[Product.CROISSANTS],
                state.order_depths[Product.JAMS],
                basket1_position,
                traderObject
            )
            basket1_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.PICNIC_BASKET1,
                state.order_depths[Product.PICNIC_BASKET1],
                basket1_fair_value,
                self.params[Product.PICNIC_BASKET1]["take_width"],
                basket1_position,
                self.params[Product.PICNIC_BASKET1]["prevent_adverse"],
                self.params[Product.PICNIC_BASKET1]["adverse_volume"],
            )
            basket1_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.PICNIC_BASKET1,
                state.order_depths[Product.PICNIC_BASKET1],
                basket1_fair_value,
                self.params[Product.PICNIC_BASKET1]["clear_width"],
                basket1_position,
                buy_order_volume,
                sell_order_volume,
            )
            basket1_make_orders, _, _ = self.make_orders(
                Product.PICNIC_BASKET1,
                state.order_depths[Product.PICNIC_BASKET1],
                basket1_fair_value,
                basket1_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.PICNIC_BASKET1]["disregard_edge"],
                self.params[Product.PICNIC_BASKET1]["join_edge"],
                self.params[Product.PICNIC_BASKET1]["default_edge"],
            )
            result[Product.PICNIC_BASKET1] = (
                basket1_take_orders + basket1_clear_orders + basket1_make_orders
            )
        
        if Product.PICNIC_BASKET2 in self.params and Product.PICNIC_BASKET2 in state.order_depths:
            basket2_position = (
                state.position[Product.PICNIC_BASKET2]
                if Product.PICNIC_BASKET2 in state.position
                else 0
            )
            basket2_fair_value = self.basket2_fair_value(
                state.order_depths[Product.PICNIC_BASKET2],
                state.order_depths[Product.CROISSANTS],
                state.order_depths[Product.JAMS],
                basket2_position,
                traderObject
            )
            basket2_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.PICNIC_BASKET2,
                state.order_depths[Product.PICNIC_BASKET2],
                basket2_fair_value,
                self.params[Product.PICNIC_BASKET2]["take_width"],
                basket2_position,
                self.params[Product.PICNIC_BASKET2]["prevent_adverse"],
                self.params[Product.PICNIC_BASKET2]["adverse_volume"],
            )
            basket2_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.PICNIC_BASKET2,
                state.order_depths[Product.PICNIC_BASKET2],
                basket2_fair_value,
                self.params[Product.PICNIC_BASKET2]["clear_width"],
                basket2_position,
                buy_order_volume,
                sell_order_volume,
            )
            basket2_make_orders, _, _ = self.make_orders(
                Product.PICNIC_BASKET2,
                state.order_depths[Product.PICNIC_BASKET2],
                basket2_fair_value,
                basket2_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.PICNIC_BASKET2]["disregard_edge"],
                self.params[Product.PICNIC_BASKET2]["join_edge"],
                self.params[Product.PICNIC_BASKET2]["default_edge"],
            )
            result[Product.PICNIC_BASKET2] = (
                basket2_take_orders + basket2_clear_orders + basket2_make_orders
            )

        volcanic_rock_voucher_products = [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500,
        ]

        if Product.VOLCANIC_ROCK in state.order_depths and all(p in state.order_depths for p in volcanic_rock_voucher_products):
            voucher_order_depths = {p: state.order_depths[p] for p in volcanic_rock_voucher_products}
            fair_values = self.volcanic_rock_voucher_fair_value(
                state.order_depths[Product.VOLCANIC_ROCK],
                voucher_order_depths,
                state.timestamp,
                traderObject
            )

            if fair_values is not None:
                for product in volcanic_rock_voucher_products:
                    order_depth = state.order_depths[product]
                    position = state.position.get(product, 0)
                    fair_value = fair_values.get(product)

                    if fair_value is None:
                        continue
                    
                    take_orders, buy_volume, sell_volume = self.take_orders(
                        product,
                        order_depth,
                        fair_value,
                        self.params[product]["take_width"],
                        position,
                        self.params[product]["prevent_adverse"],
                        self.params[product]["adverse_volume"],
                    )

                    clear_orders, buy_volume, sell_volume = self.clear_orders(
                        product,
                        order_depth,
                        fair_value,
                        self.params[product]["clear_width"],
                        position,
                        buy_volume,
                        sell_volume,
                    )

                    make_orders, _, _ = self.make_orders(
                        product,
                        order_depth,
                        fair_value,
                        position,
                        buy_volume,
                        sell_volume,
                        self.params[product]["disregard_edge"],
                        self.params[product]["join_edge"],
                        self.params[product]["default_edge"],
                    )
                    
                    result[product] = take_orders + clear_orders + make_orders

        # conversions = 0
        # if Product.MAGNIFICENT_MACARONS in self.params \
        # and Product.MAGNIFICENT_MACARONS in state.order_depths \
        # and Product.MAGNIFICENT_MACARONS in state.observations.conversionObservations:
        #     if "macarons" not in traderObject:
        #         traderObject["macarons"] = {"curr_edge": self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"], "volume_history": [], "optimized": False}
        #     macarons_position = (
        #         state.position[Product.MAGNIFICENT_MACARONS]
        #         if Product.MAGNIFICENT_MACARONS in state.position
        #         else 0
        #     )
        #     conversions = self.macarons_arb_clear(
        #         macarons_position
        #     )
        #     adap_edge = self.macarons_adap_edge(
        #         state.timestamp,
        #         traderObject["macarons"]["curr_edge"],
        #         macarons_position,
        #         traderObject,
        #     )

        #     macarons_position = 0
        #     macarons_take_orders, buy_order_volume, sell_order_volume = self.macarons_arb_take(
        #         state.order_depths[Product.MAGNIFICENT_MACARONS],
        #         state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS],
        #         adap_edge,
        #         macarons_position,
        #     )
        #     macarons_make_orders, _, _ = self.macarons_arb_make(
        #         state.order_depths[Product.MAGNIFICENT_MACARONS],
        #         state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS],
        #         macarons_position,
        #         adap_edge,
        #         buy_order_volume,
        #         sell_order_volume
        #     )
        #     result[Product.MAGNIFICENT_MACARONS] = (
        #         macarons_take_orders + macarons_make_orders
        #     )

        conversions = 0

        croissant_strategy = CroissantsStrategy(symbol=Product.CROISSANTS, limit=self.LIMIT[Product.CROISSANTS])
        if Product.CROISSANTS in state.order_depths and len(state.order_depths[Product.CROISSANTS].buy_orders) > 0 and len(state.order_depths[Product.CROISSANTS].sell_orders) > 0:
            strategy_orders, _ = croissant_strategy.run(state, traderObject)
            result[Product.CROISSANTS] = strategy_orders

        squid_ink_strategy = SquidInkStrategy(symbol=Product.SQUID_INK, limit=self.LIMIT[Product.SQUID_INK])
        if Product.SQUID_INK in state.order_depths and len(state.order_depths[Product.SQUID_INK].buy_orders) > 0 and len(state.order_depths[Product.SQUID_INK].sell_orders) > 0:
            strategy_orders, _ = squid_ink_strategy.run(state, traderObject)
            result[Product.SQUID_INK] = strategy_orders

        macarons_strategy = MacaronsStrategy(symbol=Product.MAGNIFICENT_MACARONS, limit=self.LIMIT[Product.MAGNIFICENT_MACARONS])
        if Product.MAGNIFICENT_MACARONS in state.order_depths and len(state.order_depths[Product.MAGNIFICENT_MACARONS].buy_orders) > 0 and len(state.order_depths[Product.MAGNIFICENT_MACARONS].sell_orders) > 0:
            strategy_orders, strategy_conversions = macarons_strategy.run(state, traderObject)
            result[Product.MAGNIFICENT_MACARONS] = strategy_orders
            conversions += strategy_conversions

        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData