from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List
import numpy as np
import collections
from collections import OrderedDict
import string
import copy
import json
from typing import Any
# from logger import Logger

empty_dict = {'RAINFOREST_RESIN' : 0, 'KELP' : 0}

def def_value():
    return copy.deepcopy(empty_dict)

INF = int(1e9)

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


class Trader:
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'RAINFOREST_RESIN' : 50, 'KELP' : 50}
    volume_traded = copy.deepcopy(empty_dict)
    kelp_cache = []
    kelp_dim = 4
    past_sunlight = 2500
    past_humidity = 0
    min_sun = 1400
    max_sun = 4500
    buy = False
    sell = False

    def calc_next_price_kelp(self):
        nxt_price = sum(self.kelp_cache) / len(self.kelp_cache)

        return round(nxt_price)
    
    def clear_orders(
    self,
    product: str,
    order_depth: OrderDepth,
    position: int,
    orders: List[Order],
    bid_price: float,
    ask_price: float,
) -> None:
        max_buy = self.POSITION_LIMIT[product] - position
        max_sell = self.POSITION_LIMIT[product] + position

        if position > 0:
            clearable_volume = sum(
                volume for price, volume in order_depth.buy_orders.items()
                if price >= ask_price
            )
            sell_amount = min(position, clearable_volume, max_sell)
            if sell_amount > 0:
                orders.append(Order(product, int(ask_price), -sell_amount))

        elif position < 0:
            clearable_volume = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items()
                if price <= bid_price
            )
            buy_amount = min(abs(position), clearable_volume, max_buy)
            if buy_amount > 0:
                orders.append(Order(product, int(bid_price), buy_amount))


    def compute_orders_rainforest_resin(self, product, order_depth, acc_bid, acc_ask,position, observation):
        orders: list[Order] = []

        current_position = position.get(product, 0)

        max_buy = self.POSITION_LIMIT[product] - current_position
        orders.append(Order(product, 9998, max_buy))
        max_sell = -self.POSITION_LIMIT[product] - current_position
        orders.append(Order(product, 10002, max_sell))

        return orders
    
    def compute_orders_kelp(self, product, order_depth, acc_bid, acc_ask, position, observation):
        orders: List[Order] = []

        current_position = position.get(product, 0)
        position_limit = self.POSITION_LIMIT[product]

        if len(self.kelp_cache) >= 2:
            recent_returns = [self.kelp_cache[i+1] - self.kelp_cache[i] for i in range(len(self.kelp_cache)-1)]
            vol = max(1, round(np.std(recent_returns))) 
        else:
            vol = 1

        skew = current_position / position_limit
        skew_adjustment = round(skew * vol)

        avg_price = self.calc_next_price_kelp()
        bid_price = avg_price - vol - skew_adjustment
        ask_price = avg_price + vol - skew_adjustment

        passive_bid = bid_price - 1
        passive_ask = ask_price + 1
        
        passive_buy_size = position_limit - current_position
        orders.append(Order(product, passive_bid, passive_buy_size))

        passive_sell_size = position_limit + current_position
        orders.append(Order(product, passive_ask, -passive_sell_size))
        print("TRADING KELP", current_position)
        
        # self.clear_orders(product, order_depth, current_position, orders, bid_price, ask_price)
        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask, position, observation):
        if product == "RAINFOREST_RESIN":
            # return []
            return self.compute_orders_rainforest_resin(product, order_depth[product], acc_bid, acc_ask, position, observation)
        elif product == "KELP":
            return self.compute_orders_kelp(product, order_depth[product], acc_bid, acc_ask, position, observation)
        return []
        
    
    def run(self, state: TradingState):
        logger = Logger()

        result = {'RAINFOREST_RESIN' : [], 'KELP' : []}

        if len(self.kelp_cache) == self.kelp_dim:
            self.kelp_cache.pop(0)


        if "KELP" in state.listings.keys():
            bs_kelp = max(collections.OrderedDict(sorted(state.order_depths['KELP'].sell_orders.items())))
            bb_kelp = min(collections.OrderedDict(sorted(state.order_depths['KELP'].buy_orders.items(), reverse=True)))

            self.kelp_cache.append((bs_kelp+bb_kelp)/2)


        rainforest_resin_lb = 9999
        rainforest_resin_ub = 10001
        kelp_lb = -INF
        kelp_ub = INF

        if len(self.kelp_cache) == self.kelp_dim:
            kelp_lb = self.calc_next_price_kelp()
            kelp_ub = self.calc_next_price_kelp()
        
        acc_bid = {'RAINFOREST_RESIN' : rainforest_resin_lb, 'KELP' : kelp_lb, } # we want to buy at slightly below
        acc_ask = {'RAINFOREST_RESIN' : rainforest_resin_ub, 'KELP' : kelp_ub, } # we want to sell at slightly above
        conversions = 1
        for product in state.order_depths.keys():
            order_depth: OrderDepth = state.order_depths

            if product == "RAINFOREST_RESIN" or product == "KELP":
                orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product], state.position, state.observations)
                result[product] += orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        print("SIZE", state.position.get("KELP", 0))

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
    