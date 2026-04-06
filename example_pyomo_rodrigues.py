"""
Pyomo implementation of the RTC-Tools mixed integer optimization example.

This model represents a typical setup for dewatering of lowland areas:
- Water flows from hinterland through a canal towards the sea
- If sea level < canal level: water can flow by gravity through an orifice (free)
- If sea level > canal level: water must be pumped (costs money)

Objective: Minimize total pumped water while keeping canal level below flood warning.

Reference: rtc-tools/examples/mixed_integer/
"""

import pyomo.environ as pyo
import numpy as np


def create_mixed_integer_model():
    """
    Create the Pyomo model matching the RTC-Tools mixed integer example.

    Model components from Example.mo:
    - Storage: Linear storage with A=1e6 m², H bounds [0, 0.5] m
    - Pump: Q bounds [0, 7] m³/s
    - Orifice: Q bounds [0, 10] m³/s, w=3m, d=0.8m, C=1.0
    """
    m = pyo.ConcreteModel(name="MixedIntegerOptimization")

    # =========================================================================
    # Time discretization
    # =========================================================================
    # From timeseries_import.csv: 21 time steps, hourly from 22:00 to 18:00 next day
    n_steps = 21
    dt = 3600.0  # [s] time step

    m.T = pyo.RangeSet(0, n_steps - 1)  # Time indices 0..20
    m.T_interior = pyo.RangeSet(1, n_steps - 1)  # For dynamics: 1..20

    # =========================================================================
    # Parameters from Modelica model (Example.mo)
    # =========================================================================
    # Storage properties
    A = 1.0e6  # [m²] storage area (from: storage(A=1.0e6, ...))
    H_b = 0.0  # [m] base height
    H_min = 0.0  # [m] minimum water level
    H_max = 0.5  # [m] maximum water level (flood warning)

    # Orifice properties (from path_constraints in example.py)
    w = 3.0  # [m] width of orifice
    d = 0.8  # [m] height of orifice
    C = 1.0  # [-] orifice discharge coefficient
    g = 9.8  # [m/s²] gravitational acceleration

    # Big-M constant for logical constraints
    M = 2.0  # From example.py

    # Pump and orifice flow bounds (from Example.mo)
    Q_pump_max = 7.0  # [m³/s]
    Q_orifice_max = 10.0  # [m³/s]

    # Initial state (from initial_state.csv)
    V_initial = 400000.0  # [m³]
    H_initial = V_initial / A  # = 0.4 [m]

    # =========================================================================
    # Input time series (from timeseries_import.csv)
    # =========================================================================
    # Sea level [m] - tidal pattern
    H_sea_data = {
        0: 0.0,
        1: 0.1,
        2: 0.2,
        3: 0.3,
        4: 0.4,
        5: 0.5,
        6: 0.6,
        7: 0.7,
        8: 0.8,
        9: 0.9,
        10: 1.0,
        11: 0.9,
        12: 0.8,
        13: 0.7,
        14: 0.6,
        15: 0.5,
        16: 0.4,
        17: 0.3,
        18: 0.2,
        19: 0.1,
        20: 0.0,
    }

    # Inflow discharge [m³/s] - constant
    Q_in_data = {t: 5.0 for t in range(n_steps)}

    m.H_sea = pyo.Param(m.T, initialize=H_sea_data)
    m.Q_in = pyo.Param(m.T, initialize=Q_in_data)

    # =========================================================================
    # Decision variables
    # =========================================================================
    # Storage water level [m]
    m.H_storage = pyo.Var(m.T, bounds=(H_min, H_max), initialize=H_initial)

    # Pump flow rate [m³/s]
    m.Q_pump = pyo.Var(m.T, bounds=(0.0, Q_pump_max), initialize=0.0)

    # Orifice flow rate [m³/s]
    m.Q_orifice = pyo.Var(m.T, bounds=(0.0, Q_orifice_max), initialize=0.0)

    # Binary indicator: 1 if gravity flow is possible (storage level > sea level)
    m.is_downhill = pyo.Var(m.T, domain=pyo.Binary, initialize=0)

    # =========================================================================
    # Constraints
    # =========================================================================

    # Initial condition
    m.initial_condition = pyo.Constraint(expr=m.H_storage[0] == H_initial)

    # Mass balance (forward Euler discretization)
    # dV/dt = Q_in - Q_pump - Q_orifice
    # V = A * H  =>  A * dH/dt = dV/dt = Q_in - Q_pump - Q_orifice
    # A * (H[t+1] - H[t]) / dt = Q_in[t] - Q_pump[t] - Q_orifice[t]
    def mass_balance_rule(model, t):
        return (
            A * (model.H_storage[t] - model.H_storage[t - 1])
            == dt * (model.Q_in[t - 1] - model.Q_pump[t - 1] - model.Q_orifice[t - 1])
        )

    m.mass_balance = pyo.Constraint(m.T_interior, rule=mass_balance_rule)

    # -------------------------------------------------------------------------
    # Mixed-integer constraints for orifice/gravity flow
    # These enforce: orifice flow only when storage level > sea level
    # -------------------------------------------------------------------------

    # Constraint 1: Orifice flow only when downhill
    # From RTC-Tools: Q_orifice + (1 - is_downhill) * 10 in [0, 10]
    # Equivalent to: Q_orifice <= 10 * is_downhill
    # When is_downhill=0: Q_orifice <= 0 (must be zero)
    # When is_downhill=1: Q_orifice <= 10 (normal bound)
    def orifice_downhill_only_rule(model, t):
        return model.Q_orifice[t] <= Q_orifice_max * model.is_downhill[t]

    m.orifice_downhill_only = pyo.Constraint(m.T, rule=orifice_downhill_only_rule)

    # Constraint 2: is_downhill can only be 1 if H_storage >= H_sea
    # From RTC-Tools: H_sea - H_storage - (1 - is_downhill) * M <= 0
    # When is_downhill=1: H_sea - H_storage <= 0  =>  H_storage >= H_sea (required)
    # When is_downhill=0: H_sea - H_storage <= M  (relaxed, always satisfied)
    def is_downhill_upper_rule(model, t):
        return model.H_sea[t] - model.H_storage[t] <= M * (1 - model.is_downhill[t])

    m.is_downhill_upper = pyo.Constraint(m.T, rule=is_downhill_upper_rule)

    # Constraint 3: is_downhill must be 0 if H_sea > H_storage
    # From RTC-Tools: H_sea - H_storage + is_downhill * M >= 0
    # When is_downhill=0: H_sea - H_storage >= 0  =>  H_sea >= H_storage (required when uphill)
    # When is_downhill=1: H_sea - H_storage + M >= 0  (relaxed, always satisfied)
    def is_downhill_lower_rule(model, t):
        return model.H_sea[t] - model.H_storage[t] + M * model.is_downhill[t] >= 0

    m.is_downhill_lower = pyo.Constraint(m.T, rule=is_downhill_lower_rule)

    # Constraint 4: Orifice hydraulic capacity (submerged orifice equation)
    # Q = w * C * d * sqrt(2 * g * (H_up - H_down))
    # Squared form: Q² / (w*C*d)² / (2*g) <= H_up - H_down
    # With big-M relaxation when not downhill:
    # Q² / (w*C*d)² / (2*g) + H_down - H_up <= M * (1 - is_downhill)
    #
    # H_up = H_storage (orifice.HQUp connected to storage)
    # H_down = H_sea (orifice.HQDown connected to sea level)
    K_orifice = w * C * d
    K_squared = K_orifice**2

    def orifice_capacity_rule(model, t):
        return (
            (model.Q_orifice[t] ** 2) / (K_squared * 2 * g)
            + model.H_sea[t]
            - model.H_storage[t]
            <= M * (1 - model.is_downhill[t])
        )

    m.orifice_capacity = pyo.Constraint(m.T, rule=orifice_capacity_rule)

    # =========================================================================
    # Objective: Minimize total water pumped
    # =========================================================================
    # From RTC-Tools: self.integral("Q_pump")
    # Integral approximated as sum over all time steps
    def objective_rule(model):
        return sum(model.Q_pump[t] * dt for t in model.T)

    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return m


def get_solver_executable(solver_name):
    """Find solver executable, checking IDAES bin directory if needed."""
    import shutil
    from pathlib import Path

    # Check if solver is on PATH
    if shutil.which(solver_name) or shutil.which(f"{solver_name}.exe"):
        return None  # Solver is on PATH, no need to specify

    # Try IDAES bin directory
    try:
        import idaes

        bin_dir = Path(idaes.bin_directory)
        for ext in [".exe", ""]:
            solver_path = bin_dir / f"{solver_name}{ext}"
            if solver_path.exists():
                return str(solver_path)
    except ImportError:
        pass

    return None


def solve_model(model, solver_name="couenne", tee=True):
    """
    Solve the mixed-integer nonlinear optimization problem.

    Args:
        model: Pyomo ConcreteModel
        solver_name: Solver to use ('bonmin', 'mindtpy', 'couenne', etc.)
        tee: Whether to print solver output

    Returns:
        Solver results object
    """
    if solver_name == "mindtpy":
        solver = pyo.SolverFactory("mindtpy")
        results = solver.solve(
            model, mip_solver="glpk", nlp_solver="ipopt", tee=tee, strategy="OA"
        )
    else:
        # Find solver executable
        solver_path = get_solver_executable(solver_name)

        if solver_path:
            # Create solver with explicit executable path
            solver = pyo.SolverFactory(solver_name, executable=solver_path)
        else:
            solver = pyo.SolverFactory(solver_name)

        if not solver.available():
            raise RuntimeError(
                f"Solver '{solver_name}' is not available. "
                f"Install via: pip install idaes-pse && idaes get-extensions"
            )
        results = solver.solve(model, tee=tee)

    return results


def extract_results(model):
    """Extract solution values from solved model."""
    n_steps = len(model.T)

    results = {
        "H_storage": [pyo.value(model.H_storage[t]) for t in model.T],
        "H_sea": [pyo.value(model.H_sea[t]) for t in model.T],
        "Q_pump": [pyo.value(model.Q_pump[t]) for t in model.T],
        "Q_orifice": [pyo.value(model.Q_orifice[t]) for t in model.T],
        "Q_in": [pyo.value(model.Q_in[t]) for t in model.T],
        "is_downhill": [pyo.value(model.is_downhill[t]) for t in model.T],
        "time_hours": list(range(n_steps)),
    }

    return results


def plot_results(results):
    """Plot optimization results similar to RTC-Tools output."""
    import matplotlib.pyplot as plt

    time = results["time_hours"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Water levels
    ax1 = axes[0]
    ax1.plot(time, results["H_storage"], "b-", linewidth=2, label="Storage Level")
    ax1.plot(time, results["H_sea"], "g--", linewidth=1.5, label="Sea Level")
    ax1.axhline(y=0.5, color="r", linestyle=":", label="Flood Warning")
    ax1.set_ylabel("Water Level [m]")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Mixed Integer Optimization: Pumps and Orifices")

    # Plot 2: Flow rates
    ax2 = axes[1]
    ax2.plot(time, results["Q_pump"], "r-", linewidth=2, label="Pump Flow")
    ax2.plot(time, results["Q_orifice"], "b-", linewidth=2, label="Orifice Flow")
    ax2.plot(time, results["Q_in"], "k--", linewidth=1, label="Inflow")
    ax2.set_ylabel("Flow Rate [m³/s]")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Binary indicator
    ax3 = axes[2]
    ax3.step(time, results["is_downhill"], "g-", linewidth=2, where="mid")
    ax3.fill_between(
        time, 0, results["is_downhill"], alpha=0.3, step="mid", color="green"
    )
    ax3.set_ylabel("is_downhill")
    ax3.set_xlabel("Time [hours]")
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["False", "True"])
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Creating mixed-integer optimization model...")
    model = create_mixed_integer_model()

    print("\nModel statistics:")
    print(f"  Variables: {len(list(model.component_objects(pyo.Var)))}")
    print(f"  Constraints: {len(list(model.component_objects(pyo.Constraint)))}")
    print(f"  Binary variables: {sum(1 for v in model.is_downhill.values())}")

    print("\nSolving with BONMIN...")
    try:
        results = solve_model(model, solver_name="mindtpy", tee=True)
        print(f"\nSolver status: {results.solver.status}")
        print(f"Termination condition: {results.solver.termination_condition}")

        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            sol = extract_results(model)
            total_pumped = sum(sol["Q_pump"]) * 3600  # m³
            print(f"\nTotal water pumped: {total_pumped:.0f} m³")
            print(f"Objective value: {pyo.value(model.objective):.0f} m³")

            # Print solution table
            print("\n" + "=" * 80)
            print(f"{'t':>3} {'H_sea':>8} {'H_storage':>10} {'is_down':>8} {'Q_orifice':>10} {'Q_pump':>8}")
            print("-" * 80)
            for t in range(len(sol["time_hours"])):
                print(
                    f"{t:>3} {sol['H_sea'][t]:>8.3f} {sol['H_storage'][t]:>10.4f} "
                    f"{int(sol['is_downhill'][t]):>8} {sol['Q_orifice'][t]:>10.4f} {sol['Q_pump'][t]:>8.4f}"
                )
            print("=" * 80)

            # Plot if matplotlib is available
            try:
                fig = plot_results(sol)
                import matplotlib.pyplot as plt

                plt.savefig("mixed_integer_results.png", dpi=150)
                print("\nResults saved to mixed_integer_results.png")
                plt.show()
            except ImportError:
                print("\nMatplotlib not available for plotting.")
    except RuntimeError as e:
        print(f"\nSolver error: {e}")
        print("Try installing BONMIN or use 'mindtpy' with GLPK and IPOPT.")

