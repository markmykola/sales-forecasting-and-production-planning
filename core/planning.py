# core/planning.py
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

def lp_production_plan(demand, params):

    if not isinstance(demand, pd.Series):
        raise ValueError("demand must be a pandas Series with datetime index")

    # ensure numeric floats
    demand_vals = [float(d) for d in demand.values]
    dates = list(demand.index)
    T = len(demand_vals)

    model = LpProblem("ProductionPlanning", LpMinimize)

    produce = [LpVariable(f"produce_{t}", lowBound=0) for t in range(T)]
    inventory = [LpVariable(f"inv_{t}", lowBound=0) for t in range(T)]
    backorder = [LpVariable(f"back_{t}", lowBound=0) for t in range(T)]

    prod_cost = float(params.get("prod_cost", 1.0))
    hold_cost = float(params.get("hold_cost", 0.1))
    backorder_cost = float(params.get("backorder_cost", 10.0))
    capacity = float(params.get("capacity", 1e9))
    init_inv = float(params.get("initial_inventory", 0.0))

    # objective: produce cost + holding + backorder
    model += lpSum([prod_cost * produce[t] + hold_cost * inventory[t] + backorder_cost * backorder[t] for t in range(T)])

    # constraints: capacity + inventory flow
    for t in range(T):
        model += produce[t] <= capacity
        if t == 0:
            model += init_inv + produce[t] - demand_vals[t] + backorder[t] - inventory[t] == 0
        else:
            model += inventory[t-1] + produce[t] - demand_vals[t] + backorder[t] - inventory[t] == 0

    # solve (default solver)
    model.solve()

    # collect values (float conversion)
    produce_vals = [p.value() if p.value() is not None else 0.0 for p in produce]
    inv_vals = [i.value() if i.value() is not None else 0.0 for i in inventory]
    back_vals = [b.value() if b.value() is not None else 0.0 for b in backorder]

    df = pd.DataFrame({
        "produce": produce_vals,
        "inventory_end": inv_vals,
        "backorder": back_vals,
        "demand": demand_vals
    }, index=dates)
    return df
