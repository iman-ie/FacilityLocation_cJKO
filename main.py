import numpy as np
import scipy.optimize as opt
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
# Fix random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def print_final_wasserstein_distance(P_0, best_P_k):
    """
    Prints the Wasserstein distance between the initial and optimized demand distributions.
    """
    distance = wasserstein_distance(P_0, best_P_k)
    print(f"\n📏 Final Wasserstein Distance between Initial (P_0) and Best Distribution (P_k): {distance:.4f}")



# Iterative Optimization Framework

def optimize_facility_location(I, J, T, f, a, c, d, rho, P_0, eps, eta, q_t, max_iter=15):
    """
    Solves the facility location problem under demand uncertainty using Wasserstein DRO and cJKO scheme.
    """
    import time
    start_time = time.time()

    # Initialize distribution and first-stage solution using GA
    P_k = P_0
    X, S, Y = generate_initial_solution_ga(I, J, T, q_t, P_0)

    print("\n✅ Generated Initial Solution using Genetic Algorithm:")
    print(f"X (Facility Opening Decisions):\n{X}\n")

    best_X, best_S, best_Y, best_P_k = X, S, Y, P_k
    best_obj_val = float('inf')
    second_stage_costs, Objective_Value = [], []

    for k in range(max_iter):
        # Step 1: Update demand distribution using cJKO
        best_P_k = solve_cjko(P_k, eps, c, d, rho, Y, S)

        # Step 2: Check if chance constraints are satisfied under updated P_k
        constraint_satisfied, best_P_k = check_chance_constraints_cjko(Y, S, eta, X, best_P_k, P_0, eps)

        if constraint_satisfied:
            # Step 3: Compute first-stage cost
            Z = np.zeros_like(X)
            Z[:, 0] = X[:, 0]
            Z[:, 1:] = np.logical_and(X[:, 1:], np.logical_not(X[:, :-1])).astype(int)

            term1 = np.sum(f * Z + a * (S - np.roll(S, shift=1, axis=1)))

            # Step 4: Compute second-stage cost under updated distribution
            expected_cost = np.mean([
                np.sum(c * (Y * best_P_k[scenario][:, np.newaxis])) +
                np.sum(rho[np.newaxis, :, np.newaxis] *
                       np.maximum(0, Y * best_P_k[scenario][:, np.newaxis] - S[:, np.newaxis, :]))
                for scenario in range(best_P_k.shape[0])
            ])

            penalty = np.sum(np.all(X == 0, axis=0)) * 1e6
            obj_val = term1 + expected_cost + penalty

            second_stage_costs.append(expected_cost)
            Objective_Value.append(obj_val)

            print(
                f"Iteration {k}: First-Stage Cost = {term1:.4f}, "
                f"Second-Stage Expected Cost = {expected_cost:.4f}, "
                f"Total Objective Value = {obj_val:.4f}, Epsilon = {eps:.4f}"
            )

            # Step 5: Keep best solution
            if obj_val < best_obj_val:
                best_obj_val = obj_val
                best_X, best_S, best_Y, best_P_k = X, S, Y, best_P_k

            # Step 6: Update S and Y (with random adjustment)
            S = np.maximum(S + np.random.randint(-2, 3, size=S.shape), 0) * X
            Y = np.minimum(Y, S[:, np.newaxis, :])

        else:
            print(f"Iteration {k}: ❌ Constraint Violated — Epsilon Remains at {eps}")

        # Step 7: Track chance constraint satisfaction
        _, best_P_k = check_chance_constraints_cjko(Y, S, eta, X, best_P_k, P_0, eps)
        demand_satisfaction = calculate_demand_satisfaction(Y, S, X, best_P_k)
        print(f"Iteration {k}: Demand Satisfaction = {demand_satisfaction:.2f}%")

    # Final summary
    end_time = time.time()
    print(f"\n⏱️ Optimization Completed in {end_time - start_time:.2f} seconds")
    print("\n✅ Best Optimized Solution:")
    print(f"Best Objective Value: {best_obj_val:.4f}")
    print(f"X (Facility Decisions):\n{best_X}")
    print(f"S (Storage Levels):\n{best_S}")
    print(f"Y (Allocations):\n{best_Y}")

    return best_X, best_S, best_Y, best_P_k
