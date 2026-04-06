#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: martijnkrikke
"""

import pyomo.environ as pyo
import pandas as pd
import random

random.seed(43)
# 1. Data inladen
time_series = pd.read_csv('timeseries_import.csv')

model = pyo.ConcreteModel()
model.T = pyo.RangeSet(1, 21)

# --- Parameters ---
h_sea_dict = {i+1: val for i, val in enumerate(time_series["H_sea"][:21])}
model.H_sea = pyo.Param(model.T, initialize=h_sea_dict)
q_in_dict = {i+1: val  for i, val in enumerate(time_series["Q_in"][:21])}
model.Q_in = pyo.Param(model.T, initialize=q_in_dict)
energy_dict = {i+1: val for i, val in enumerate(time_series["Energy"][:21])}
model.Energy = pyo.Param(model.T, initialize=energy_dict)

M = 2
STORAGE_MAX = 0.4 + random.uniform(0, 0.2)
H_start = STORAGE_MAX - 0.1
A = 1e6
dt = 3600
PUMP_MAX = 3 - random.uniform(0, 1)

# --- Variables ---
model.Q_orifice = pyo.Var(model.T, domain=pyo.NonNegativeReals)
model.Q_pump = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, PUMP_MAX))
model.x = pyo.Var(model.T, domain=pyo.Binary)
model.H_storage = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, STORAGE_MAX))

# --- Constraints ---
def storage_balance_rule(model, t):
    if t == 1:
        return model.H_storage[t] == H_start
    else:
        return (A * (model.H_storage[t] - model.H_storage[t - 1])
                == dt * (model.Q_in[t - 1] - model.Q_pump[t - 1] - model.Q_orifice[t - 1]))
model.storage_balance = pyo.Constraint(model.T, rule=storage_balance_rule)

def only_downhill_rule1(model, t):
    return model.Q_orifice[t] + (1 - model.x[t]) * 10 >= 0
model.only_downhill1 = pyo.Constraint(model.T, rule=only_downhill_rule1)

def only_downhill_rule2(model, t):
    return model.Q_orifice[t] + (1 - model.x[t]) * 10  <= 10
model.only_downhill2 = pyo.Constraint(model.T, rule=only_downhill_rule2)

def fix_downhill_rule1(model, t):
    return model.H_sea[t] - model.H_storage[t] - (1 - model.x[t]) * M <= 0
model.fix_downhill1 = pyo.Constraint(model.T, rule=fix_downhill_rule1)

def fix_downhill_rule2(model, t):
    return -model.H_sea[t] + model.H_storage[t] - model.x[t] * M <= 0
model.fix_downhill2 = pyo.Constraint(model.T, rule=fix_downhill_rule2)

def fix_downhill_rule2(model, t):
    return -model.H_sea[t] + model.H_storage[t] - model.x[t] * M <= 0
model.fix_downhill2 = pyo.Constraint(model.T, rule=fix_downhill_rule2)

w = 3.0  # m       width of orifice
d = 0.8  # m       hight of orifice
C = 1.0  # none    orifice constant
g = 9.8  # m/s^2   gravitational acceleration
def torricelli_law_rule(model, t):
    return ((model.Q_orifice[t] / (w * C * d)) ** 2) / (2 * g) + model.H_sea[t] - model.H_storage[t] - (1 - model.x[t]) * M <= 0
model.torricelli_law = pyo.Constraint(model.T, rule=torricelli_law_rule)

# --- Objective ---
model.obj = pyo.Objective(expr=sum(model.Q_pump[t] * dt for t in model.T), sense=pyo.minimize)

solver = pyo.SolverFactory('gurobi') 
results = solver.solve(model, tee=True, symbolic_solver_labels=True)
# For mindtpy: results = solver.solve(model, mip_solver="glpk", nlp_solver="ipopt", tee=True, strategy="OA")

if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
    solver.options['ResultFile'] = 'example_model_iis.ilp'
    solver.solve(model, symbolic_solver_labels=True)

if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    results_list = []
    time_col = time_series["UTC"].iloc[:21].tolist()    
    for t in model.T:
        t_index = t - 1 
        begin_time = "2013-05-19 22:00:00"
        try:
            q_ori = pyo.value(model.Q_orifice[t])
            q_pmp = pyo.value(model.Q_pump[t])
            is_dw = pyo.value(model.x[t])
            s_lvl = pyo.value(model.H_sea[t])
            h_str = pyo.value(model.H_storage[t])
        except ValueError:
            q_ori, q_pmp, is_dw, s_lvl, h_str = 0.0, 0.0, 0.0, 0.0, 0.0
    
        results_list.append({
            'time': time_col[t_index],
            'Q_orifice': q_ori,
            'Q_pump': q_pmp,
            'is_downhill': is_dw,
            'sea_level': s_lvl,
            'storage_level': h_str
        })
    
    
    df_output = pd.DataFrame(results_list)
    
    output_path = '/Users/martijnkrikke/Documents/Scriptie/mixed_integer/output_results.csv'
    
    df_output.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S', float_format='%.6f') 
    
    print(df_output[["Q_orifice", "Q_pump", "is_downhill", "storage_level"]])
















