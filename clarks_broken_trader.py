from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import collections
from collections import OrderedDict
import string
import copy
from logger import Logger

empty_dict = {'RAINFOREST_RESIN' : 0, 'KELP' : 0, 'ORCHIDS':0, 'CHOCOLATE':0, 'STRAWBERRIES':0, 'ROSES':0,'GIFT_BASKET':0}

def def_value():
    return copy.deepcopy(empty_dict)

INF = int(1e9)




class Trader:
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'RAINFOREST_RESIN' : 50, 'KELP' : 50, 'ORCHIDS': 100, 'CHOCOLATE': 250 , 'STRAWBERRIES':350, 'ROSES':60,'GIFT_BASKET':60}
    volume_traded = copy.deepcopy(empty_dict)
    kelp_cache = []
    kelp_dim = 15
    past_sunlight = 2500
    past_humidity = 0
    min_sun = 1400
    max_sun = 4500
    buy = False
    sell = False
    
    def calc_hours(self, cur_sun):
        return (cur_sun - 1400)*24/(4500-1400)



    def calc_next_price_kelp(self):
        # bananas cache stores price from 1 day ago, current day resp
        # by price, here we mean mid price

        # LR with MID_PRICE
        # LR -- dim 4

        

        # coef = [0.20576273, 0.20496576, 0.26110132, 0.32726722]
        # intercept = 1.8296480069286645
        # LR -- dim 3
        # coef = [0.28429297, 0.32881363, 0.3858157 ]
        # intercept = 2.182763901444787

       

        # LR -- dim 5
        # coef = [0.14029913, 0.14102833, 0.17114791, 0.23196233, 0.31514739]
        # intercept = 2.0945270917454764
        # LR -- dim 10
        # coef = [0.02341998, 0.02900498, 0.0108657, 0.04246118, 0.06291289, 0.087481, 0.10168514, 0.14143746, 0.20766892, 0.2927222]
        # intercept = 1.7209784733086053

        # LR with bestbuy + bestsell / 2
        # LR -- dim 4
        

        coef = [0.06046632, 0.05756016, 0.17241528, 0.70930228]
        intercept = 0.5195663471117768
        # LR -- dim 3
        # coef = [0.10085063, 0.18358259, 0.71530347]
        # intercept = 0.53438856279422

        
        # LR -- dim 5
        # coef = [0.0033999, 0.01865361, 0.03401545, 0.18314122, 0.76065801]
        # intercept = 0.6662652289442121

        # nxt_price = intercept
        # for i, val in enumerate(self.kelp_cache):
        #     nxt_price += val * coef[i]

        nxt_price = sum(self.kelp_cache) / len(self.kelp_cache)

        return round(nxt_price)


    def compute_orders_rainforest_resin(self, product, order_depth, acc_bid, acc_ask,position, observation):
        orders: list[Order] = []

        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_position = position.get(product, 0)

        max_buy = self.POSITION_LIMIT[product] - current_position
        orders.append(Order(product, 9998, max_buy))
        max_sell = -self.POSITION_LIMIT[product] - current_position
        orders.append(Order(product, 10002, max_sell))

        return orders
    
    def compute_orders_kelp(self, product, order_depth, acc_bid, acc_ask, position, observation):
        orders: list[Order] = []

        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        current_position = position.get(product, 0)
        for ask, vol in sell_orders.items():
            if (ask <= acc_bid)and current_position < self.POSITION_LIMIT[product]:
                order_for = min(-vol, self.POSITION_LIMIT[product] - current_position)
                current_position += order_for
                orders.append(Order(product, ask, order_for))

        for bid, vol in buy_orders.items():
            if (bid >= acc_ask) and current_position > -self.POSITION_LIMIT[product]:
                order_for = max(-vol, -self.POSITION_LIMIT[product]-current_position)
                current_position += order_for
                orders.append(Order(product, bid, order_for))
        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask, position, observation):
        if product == "RAINFOREST_RESIN":
            return []
            return self.compute_orders_rainforest_resin(product, order_depth[product], acc_bid, acc_ask, position, observation)
        elif product == "KELP":
            return self.compute_orders_kelp(product, order_depth[product], acc_bid, acc_ask, position, observation)
        return []
        
    
    def run(self, state: TradingState):
        logger = Logger()

        result = {'RAINFOREST_RESIN' : [], 'KELP' : [], 'ORCHIDS': []}

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
            kelp_lb = self.calc_next_price_kelp()- 1
            kelp_ub = self.calc_next_price_kelp()+ 1
        
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

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
    