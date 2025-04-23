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

# ------------------------------


# ------------------------------



# ------------------------------
#  Solve cJKO Wasserstein-check-chance-constrained update
# ------------------------------
def check_chance_constraints_cjko(Y, S, eta, X, P_k, P_0, eps, max_iter=10, tol=1e-6):
    """
    Enforces chance constraints under Wasserstein ambiguity by checking
    the worst-case violation probability within the ambiguity set.

    Parameters:
        Y (ndarray): Allocation matrix (J x I x T)
        S (ndarray): Storage matrix (J x T)
        eta (float): Probability confidence level (e.g., 0.95)
        X (ndarray): Facility decisions (J x T), not directly used here
        P_k (ndarray): Current demand distribution (num_scenarios x I)
        P_0 (ndarray): Nominal (initial) distribution
        eps (float): Wasserstein radius
        max_iter (int): Maximum optimization iterations
        tol (float): Optimization tolerance

    Returns:
        tuple: (constraint_satisfied (bool), best_P_k (ndarray))
    """

    num_scenarios, num_customers = P_k.shape
    v = np.ones((num_customers,)) / np.sqrt(num_customers)

    def build_rotation_matrix(alpha, d):
        R = np.eye(d)
        for l in range(d - 1):
            i, j = l, l + 1
            theta = alpha[l]
            G = np.eye(d)
            G[i, i] = np.cos(theta)
            G[i, j] = -np.sin(theta)
            G[j, i] = np.sin(theta)
            G[j, j] = np.cos(theta)
            R = R @ G
        return R

    def chance_violation_prob(P):
        """Computes the empirical probability of violating storage constraint under demand."""
        violations = 0
        for scenario in range(num_scenarios):
            demand = P[scenario]
            for t in range(Y.shape[2]):
                for j in range(Y.shape[0]):
                    unmet_demand = np.sum(Y[j, :, t] * demand) - S[j, t]
                    if unmet_demand > 0:
                        violations += 1
                        break  # Stop checking this scenario
                else:
                    continue
                break
        return violations / num_scenarios

    best_P_k = P_k.copy()
    best_violation = 0.0

    for k in range(max_iter):
        def wasserstein_obj(params):
            """Objective: maximize the chance constraint violation."""
            lambda_k = params[:num_scenarios]
            alpha_k = params[num_scenarios:].reshape(num_scenarios, num_customers)

            P_new = np.zeros_like(P_k)
            for i in range(num_scenarios):
                R_i = build_rotation_matrix(alpha_k[i], num_customers)
                direction = v @ R_i
                P_new[i] = P_k[i] - lambda_k[i] * direction

            return chance_violation_prob(np.clip(P_new, 0, None))

        def apply_transport(P_k, lambda_k, alpha_k, v):
            """Applies transport updates with projection."""
            P_new = np.zeros_like(P_k)
            for i in range(P_k.shape[0]):
                R_i = build_rotation_matrix(alpha_k[i], P_k.shape[1])
                direction = v @ R_i
                P_new[i] = P_k[i] - lambda_k[i] * direction
            return np.clip(P_new, 0, None)

        def wasserstein_constraint(params):
            """Ensures the Wasserstein distance remains within ε."""
            lambda_k = params[:num_scenarios]
            alpha_k = params[num_scenarios:].reshape(num_scenarios, num_customers)
            P_new = apply_transport(P_k, lambda_k, alpha_k, v)
            distance = np.mean([np.linalg.norm(P_new[i] - P_k[i])**2 for i in range(num_scenarios)])
            return eps - distance

        # === Optimization Setup ===
        initial_params = np.hstack([
            np.random.uniform(0.5, 1.0, num_scenarios),
            np.random.uniform(-np.pi, np.pi, num_scenarios * num_customers)
        ])

        result = opt.minimize(
            fun=lambda x: wasserstein_obj(x),
            x0=initial_params,
            constraints=[{'type': 'ineq', 'fun': wasserstein_constraint}],
            method='COBYLA',
            tol=tol
        )

        lambda_k = result.x[:num_scenarios]
        alpha_k = result.x[num_scenarios:].reshape(num_scenarios, num_customers)
        P_new = apply_transport(P_k, lambda_k, alpha_k, v)
        P_new = P_new / np.sum(P_new, axis=1, keepdims=True)  # Normalize to maintain valid demand proportions

        current_violation = chance_violation_prob(P_new)

        if current_violation > best_violation:
            best_violation = current_violation
            best_P_k = P_new.copy()

        if best_violation >= (1 - eta):
            break

    constraint_satisfied = best_violation <= (1 - eta)
    return constraint_satisfied, best_P_k

def print_final_wasserstein_distance(P_0, best_P_k):
    """
    Prints the Wasserstein distance between the initial and optimized demand distributions.
    """
    distance = wasserstein_distance(P_0, best_P_k)
    print(f"\n📏 Final Wasserstein Distance between Initial (P_0) and Best Distribution (P_k): {distance:.4f}")


def calculate_demand_satisfaction(Y, S, X, P):
    """
    Calculates the percentage of scenarios where all demands are satisfied without exceeding storage.
    """
    S_expanded = np.expand_dims(S, axis=1)  # (J, 1, T) → for broadcasting
    feasible_Y = np.minimum(Y, S_expanded) * X[:, np.newaxis, :]  # Feasible allocation

    # Count how many time periods violate the condition (Y > S)
    violating_scenarios = np.sum([
        np.any(feasible_Y[:, :, t] > S_expanded[:, :, t]) for t in range(S.shape[1])
    ])
    total_scenarios = P.shape[0]

    return (1 - violating_scenarios / total_scenarios) * 100  # as percentage


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
