from typing import Any, List
import json
import jsonpickle
import numpy as np
import math

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = "KELP"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"


BASKET1_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}

BASKET2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}

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
        "deviation_threshold": 0.005,
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
        "deviation_threshold": 0.03,
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
        "synthetic_weight": 0.05,
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
        "synthetic_weight": 0.02,
        "volatility_window_size": 10,
        "adverse_volatility": 0.1,
    },
}


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

            if "kelp_price_history" not in traderObject:
                traderObject["kelp_price_history"] = []
            traderObject["kelp_price_history"].append(mmmid_price)
            window_size = self.params[Product.KELP]["window_size"]
            price_history = traderObject["kelp_price_history"][-window_size:]
            traderObject["kelp_price_history"] = price_history

            if len(price_history) == window_size:
                mean_price = sum(price_history) / len(price_history)
                deviation = (mmmid_price - mean_price) / mean_price
                deviation_threshold = self.params[Product.KELP]["deviation_threshold"]
                if abs(deviation) > deviation_threshold:
                    fair = mmmid_price - (deviation * mean_price * 0.5)
                else:
                    fair = mmmid_price
            else:
                fair = mmmid_price

            return fair
        return None
    
    def squid_ink_fair_value(self, order_depth: OrderDepth, position: int, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                mmmid_price = (best_ask + best_bid) / 2
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if "squid_ink_price_history" not in traderObject:
                traderObject["squid_ink_price_history"] = []
            traderObject["squid_ink_price_history"].append(mmmid_price)
            window_size = self.params[Product.SQUID_INK]["window_size"]
            price_history = traderObject["squid_ink_price_history"][-window_size:]
            traderObject["squid_ink_price_history"] = price_history

            if len(price_history) == window_size:
                sorted_price_history = sorted(price_history)

                # take middle 50%
                lower_index = window_size // 4
                upper_index = window_size - lower_index
                middle_prices = sorted_price_history[lower_index:upper_index]  
                
                truncated_mean_price = sum(middle_prices) / len(middle_prices)

                deviation = (mmmid_price - truncated_mean_price) / truncated_mean_price
                deviation_threshold = self.params[Product.SQUID_INK]["deviation_threshold"]
                
                if abs(deviation) > deviation_threshold:
                    return truncated_mean_price
                elif position != 0:
                    return mmmid_price
                else:
                    return None
            elif position != 0:
                return mmmid_price
            else:
                return None
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

    def croissants_fair_value(
        self,
        order_depth: OrderDepth,
        basket2_order_depth: OrderDepth,
        jams_order_depth: OrderDepth,
        position: int,
        traderObject
    ) -> float:
        mid = self.filtered_mid(Product.CROISSANTS, order_depth)
        basket2_mid = self.filtered_mid(Product.PICNIC_BASKET2, basket2_order_depth)
        jams_mid = self.filtered_mid(Product.JAMS, jams_order_depth)

        if mid is None:
            return None
        if basket2_mid is None or jams_mid is None:
            return mid
        
        croissants_synthetic_mid = (basket2_mid - (jams_mid * BASKET2_WEIGHTS[Product.JAMS])) / BASKET2_WEIGHTS[Product.CROISSANTS]

        synthetic_weight = self.params[Product.CROISSANTS]["synthetic_weight"]
        if abs(position) > self.LIMIT[Product.CROISSANTS] * 0.7:
            synthetic_weight *= 0.5
        if abs(position) == self.LIMIT[Product.CROISSANTS]:
            synthetic_weight = 0
        return (1 - synthetic_weight) * mid + synthetic_weight * croissants_synthetic_mid

    def jams_fair_value(
        self,
        order_depth: OrderDepth,
        basket2_order_depth: OrderDepth,
        croissants_order_depth: OrderDepth,
        position: int,
        traderObject
    ) -> float:
        mid = self.filtered_mid(Product.JAMS, order_depth)
        basket2_mid = self.filtered_mid(Product.PICNIC_BASKET2, basket2_order_depth)
        croissants_mid = self.filtered_mid(Product.CROISSANTS, croissants_order_depth)

        if mid is None:
            return None
        if basket2_mid is None or croissants_mid is None:
            return mid
        
        jams_synthetic_mid = (basket2_mid - (croissants_mid * BASKET2_WEIGHTS[Product.CROISSANTS])) / BASKET2_WEIGHTS[Product.JAMS]

        synthetic_weight = self.params[Product.JAMS]["synthetic_weight"]
        if abs(position) > self.LIMIT[Product.JAMS] * 0.7:
            synthetic_weight *= 0.5
        if abs(position) == self.LIMIT[Product.JAMS]:
            synthetic_weight = 0
        return (1 - synthetic_weight) * mid + synthetic_weight * jams_synthetic_mid

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

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            squid_ink_fair_value = self.squid_ink_fair_value(
                state.order_depths[Product.SQUID_INK], squid_ink_position, traderObject
            )
            if squid_ink_fair_value is not None:
                squid_ink_take_orders, buy_order_volume, sell_order_volume = (
                    self.take_orders(
                        Product.SQUID_INK,
                        state.order_depths[Product.SQUID_INK],
                        squid_ink_fair_value,
                        self.params[Product.SQUID_INK]["take_width"],
                        squid_ink_position,
                        self.params[Product.SQUID_INK]["prevent_adverse"],
                        self.params[Product.SQUID_INK]["adverse_volume"],
                    )
                )
                squid_ink_clear_orders, buy_order_volume, sell_order_volume = (
                    self.clear_orders(
                        Product.SQUID_INK,
                        state.order_depths[Product.SQUID_INK],
                        squid_ink_fair_value,
                        self.params[Product.SQUID_INK]["clear_width"],
                        squid_ink_position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                )
                squid_ink_make_orders, _, _ = self.make_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squid_ink_fair_value,
                    squid_ink_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.SQUID_INK]["disregard_edge"],
                    self.params[Product.SQUID_INK]["join_edge"],
                    self.params[Product.SQUID_INK]["default_edge"],
                )
                result[Product.SQUID_INK] = (
                    squid_ink_take_orders + squid_ink_clear_orders + squid_ink_make_orders
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

        # if Product.CROISSANTS in self.params and Product.CROISSANTS in state.order_depths:
        #     croissants_position = (
        #         state.position[Product.CROISSANTS]
        #         if Product.CROISSANTS in state.position
        #         else 0
        #     )
        #     croissants_fair_value = self.croissants_fair_value(
        #         state.order_depths[Product.CROISSANTS],
        #         state.order_depths[Product.PICNIC_BASKET2],
        #         state.order_depths[Product.JAMS],
        #         croissants_position,
        #         traderObject
        #     )
        #     croissants_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
        #         Product.CROISSANTS,
        #         state.order_depths[Product.CROISSANTS],
        #         croissants_fair_value,
        #         self.params[Product.CROISSANTS]["take_width"],
        #         croissants_position,
        #         self.params[Product.CROISSANTS]["prevent_adverse"],
        #         self.params[Product.CROISSANTS]["adverse_volume"],
        #     )
        #     croissants_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
        #         Product.CROISSANTS,
        #         state.order_depths[Product.CROISSANTS],
        #         croissants_fair_value,
        #         self.params[Product.CROISSANTS]["clear_width"],
        #         croissants_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #     )
        #     croissants_make_orders, _, _ = self.make_orders(
        #         Product.CROISSANTS,
        #         state.order_depths[Product.CROISSANTS],
        #         croissants_fair_value,
        #         croissants_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.CROISSANTS]["disregard_edge"],
        #         self.params[Product.CROISSANTS]["join_edge"],
        #         self.params[Product.CROISSANTS]["default_edge"],
        #     )
        #     result[Product.CROISSANTS] = (
        #         croissants_take_orders + croissants_clear_orders + croissants_make_orders
        #     )
        
        # if Product.JAMS in self.params and Product.JAMS in state.order_depths:
        #     jams_position = (
        #         state.position[Product.JAMS]
        #         if Product.JAMS in state.position
        #         else 0
        #     )
        #     jams_fair_value = self.jams_fair_value(
        #         state.order_depths[Product.JAMS],
        #         state.order_depths[Product.PICNIC_BASKET2],
        #         state.order_depths[Product.CROISSANTS],
        #         jams_position,
        #         traderObject
        #     )
        #     jams_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
        #         Product.JAMS,
        #         state.order_depths[Product.JAMS],
        #         jams_fair_value,
        #         self.params[Product.JAMS]["take_width"],
        #         jams_position,
        #         self.params[Product.JAMS]["prevent_adverse"],
        #         self.params[Product.JAMS]["adverse_volume"],
        #     )
        #     jams_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
        #         Product.JAMS,
        #         state.order_depths[Product.JAMS],
        #         jams_fair_value,
        #         self.params[Product.JAMS]["clear_width"],
        #         jams_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #     )
        #     jams_make_orders, _, _ = self.make_orders(
        #         Product.JAMS,
        #         state.order_depths[Product.JAMS],
        #         jams_fair_value,
        #         jams_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.JAMS]["disregard_edge"],
        #         self.params[Product.JAMS]["join_edge"],
        #         self.params[Product.JAMS]["default_edge"],
        #     )
        #     result[Product.JAMS] = (
        #         jams_take_orders + jams_clear_orders + jams_make_orders
        #     )

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData