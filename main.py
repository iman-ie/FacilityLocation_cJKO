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
# Genetic Algorithm Initialization for Facility Location
# ------------------------------
def generate_initial_solution_ga(I, J, T, q_t, P_0, pop_size=50, generations=100, mutation_rate=0.1):
    """
    Generates an initial solution using a Genetic Algorithm (GA) for the facility location problem.
    
    Parameters:
        I (int): Number of customers
        J (int): Number of facilities
        T (int): Number of time periods
        q_t (array): Capacity per facility per time period
        P_0 (array): Initial demand scenarios
        pop_size (int): GA population size
        generations (int): Number of generations
        mutation_rate (float): Mutation rate

    Returns:
        X (array): Facility opening decisions
        S (array): Storage levels
        Y (array): Allocation decisions
    """
    def fitness(individual):
        # Decode the individual into decision variables
        X, S, Y = decode_solution(individual, J, T, I, q_t)

        # Compute first-stage costs
        Z = np.zeros_like(X)
        Z[:, 0] = X[:, 0]
        Z[:, 1:] = np.logical_and(X[:, 1:], np.logical_not(X[:, :-1])).astype(int)
        inventory_cost = np.sum(a * (S - np.roll(S, shift=1, axis=1)))
        term1 = np.sum(f * Z) + inventory_cost

        # Compute expected second-stage cost over scenarios
        expected_cost = np.mean([
            np.sum(c * (Y * P_0[scenario][:, np.newaxis])) +
            np.sum(rho[np.newaxis, :, np.newaxis] *
                   np.maximum(0, Y * P_0[scenario][:, np.newaxis] - S[:, np.newaxis, :]))
            for scenario in range(P_0.shape[0])
        ])

        # Total cost (negative for minimization)
        cost = term1 + expected_cost
        return -cost,

    def decode_solution(individual, J, T, I, q_t):
        # Extract and reshape decision variables from GA individual
        X = np.random.choice([0, 1], size=(J, T), p=[0.5, 0.5])
        for t in range(T):  # Ensure at least one facility is open per period
            if np.sum(X[:, t]) == 0:
                X[np.random.randint(0, J), t] = 1

        # Storage levels and allocations
        S = np.clip(np.array(individual[J * T:2 * J * T]).reshape(J, T), 3, 15) * X
        Y = np.clip(np.array(individual[2 * J * T:]).reshape(J, I, T), 0, 1) * X[:, np.newaxis, :]
        Y = np.minimum(Y, S[:, np.newaxis, :])
        return X, S, Y

    # Register GA components
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float,
                     n=(J * T + J * T + J * I * T))  # [X, S, Y]
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness)

    # Run GA
    pop = toolbox.population(n=pop_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)
    best_ind = tools.selBest(pop, k=1)[0]

    return decode_solution(best_ind, J, T, I, q_t)


# ------------------------------
# Step 2: Compute Wasserstein Distance Constraint
# ------------------------------
def wasserstein_distance(P, P_k):
    """
    Computes the squared 2-Wasserstein distance between two sets of discrete probability distributions.

    Parameters:
        P (ndarray): Updated demand distribution (shape: scenarios × customers).
        P_k (ndarray): Previous demand distribution (shape: scenarios × customers).

    Returns:
        float: Mean squared 2-norm (Wasserstein distance squared) between corresponding scenarios.
    """
    return np.mean([
        np.linalg.norm(P[i] - P_k[i], ord=2) ** 2
        for i in range(P.shape[0])
    ])


def solve_cjko(P_k, eps, c, d, rho, Y, S, max_iter=10, tol=1e-6):
    if P_k.ndim == 1:
        P_k = P_k.reshape(1, -1)
    num_scenarios, num_customers = P_k.shape
    v = np.ones((num_customers,)) / np.sqrt(num_customers)

    second_stage_costs = []
    best_P_k = P_k.copy()
    best_second_stage_cost = -float('inf')

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

    for k in range(max_iter):

        def wasserstein_obj(params):
            lambda_k = params[:num_scenarios]
            alpha_k = params[num_scenarios:].reshape(num_scenarios, num_customers - 1)
            P_new = np.zeros_like(P_k)
            for i in range(num_scenarios):
                R_i = build_rotation_matrix(alpha_k[i], num_customers)
                direction = v @ R_i
                P_new[i] = P_k[i] - lambda_k[i] * direction
            expected_cost = np.mean([
                np.sum(c * (Y * P_new[scenario][:, np.newaxis])) +
                np.sum(rho[np.newaxis, :, np.newaxis] *
                       np.maximum(0, Y * P_new[scenario][:, np.newaxis] - S[:, np.newaxis, :]))
                for scenario in range(P_new.shape[0])
            ])

            return expected_cost

        def apply_transport(P_k, lambda_k, alpha_k, v):
            P_new = np.zeros_like(P_k)
            for i in range(P_k.shape[0]):
                R_i = build_rotation_matrix(alpha_k[i], P_k.shape[1])
                direction = v @ R_i
                P_new[i] = P_k[i] - lambda_k[i] * direction
            return np.clip(P_new, 0, None)

        def wasserstein_constraint(params):
            lambda_k = params[:num_scenarios]
            alpha_k = params[num_scenarios:].reshape(num_scenarios, num_customers - 1)
            P_new = apply_transport(P_k, lambda_k, alpha_k, v)
            return eps - wasserstein_distance(P_new, P_k)

        constraints = [{'type': 'ineq', 'fun': wasserstein_constraint}]
        initial_params = np.hstack([
            np.random.uniform(0.5, 1.0, num_scenarios),
            np.random.uniform(-np.pi, np.pi, num_scenarios * (num_customers - 1))
        ])

        result = opt.minimize(lambda x: -wasserstein_obj(x), initial_params,
                              constraints=constraints, tol=tol, method='COBYLA')

        lambda_k = result.x[:num_scenarios]
        alpha_k = result.x[num_scenarios:].reshape(num_scenarios, num_customers - 1)

        P_new = apply_transport(P_k, lambda_k, alpha_k, v)

        P_new = np.clip(P_new, 0, None)


        expected_cost = np.mean([
            np.sum(c * (Y * P_new[scenario][:, np.newaxis])) +
            np.sum(rho[np.newaxis, :, np.newaxis] *
                   np.maximum(0, Y * P_new[scenario][:, np.newaxis] - S[:, np.newaxis, :]))
            for scenario in range(P_new.shape[0])
        ])

        second_stage_costs.append(expected_cost)

        if expected_cost > best_second_stage_cost:
            best_second_stage_cost = expected_cost
            best_P_k = P_new.copy()

        wasserstein_dist = wasserstein_distance(P_new, P_k)
        print(f"cJKO Iteration {k}: Second-Stage Cost = {expected_cost:.4f} Wasserstein Distance: {wasserstein_dist:.4f}")

        P_k = P_new.copy()

    print("\nUpdated Demand Scenarios from Best P_k:")
    for i in range(min(6, best_P_k.shape[0])):
        print(f"Scenario {i}: {best_P_k[i]}")

    #plt.plot(range(max_iter), second_stage_costs, marker='o', linestyle='-')
    #plt.xlabel('cJKO Iteration')
    #plt.ylabel('Second-Stage Cost')
    #plt.title('Maximization of Second-Stage Cost During cJKO')
    #plt.grid(True)
    #plt.show()

    return np.clip(best_P_k, 0, np.max(best_P_k) * 1.2)



def check_chance_constraints_cjko(Y, S, eta, X, P_k, P_0, eps, max_iter=10, tol=1e-6):
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
        violation_count = 0
        for scenario in range(num_scenarios):
            demand = P[scenario]
            # Check if any facility runs out of stock under this demand scenario
            violated = False
            for t in range(Y.shape[2]):
                for j in range(Y.shape[0]):
                    unmet_demand = np.sum(Y[j, :, t] * demand) - S[j, t]
                    if unmet_demand > 0:
                        violated = True
                        break
                if violated:
                    break
            if violated:
                violation_count += 1
        return violation_count / num_scenarios

    best_P_k = P_k.copy()
    best_violation = 0

    for k in range(max_iter):
        def wasserstein_obj(params):
            lambda_k = params[:num_scenarios]
            alpha_k = params[num_scenarios:].reshape(num_scenarios, num_customers)
            P_new = np.zeros_like(P_k)
            for i in range(num_scenarios):
                R_i = build_rotation_matrix(alpha_k[i], num_customers)
                direction = v @ R_i
                P_new[i] = P_k[i] - lambda_k[i] * direction
            return chance_violation_prob(P_new)

        def apply_transport(P_k, lambda_k, alpha_k, v):
            P_new = np.zeros_like(P_k)
            for i in range(P_k.shape[0]):
                R_i = build_rotation_matrix(alpha_k[i], P_k.shape[1])
                direction = v @ R_i
                P_new[i] = P_k[i] - lambda_k[i] * direction
            return np.clip(P_new, 0, None)


        def wasserstein_constraint(params):
            lambda_k = params[:num_scenarios]
            alpha_k = params[num_scenarios:].reshape(num_scenarios, num_customers)
            P_new = apply_transport(P_k, lambda_k, alpha_k, v)
            return eps - wasserstein_distance(P_new, P_k)

        constraints = [{'type': 'ineq', 'fun': wasserstein_constraint}]
        initial_params = np.hstack([
            np.random.uniform(0.5, 1.0, num_scenarios),
            np.random.uniform(-np.pi, np.pi, num_scenarios * num_customers)
        ])

        result = opt.minimize(lambda x: wasserstein_obj(x), initial_params, constraints=constraints, tol=tol, method='COBYLA')

        lambda_k = result.x[:num_scenarios]
        alpha_k = result.x[num_scenarios:].reshape(num_scenarios, num_customers)

        P_new = apply_transport(P_k, lambda_k, alpha_k, v)
        P_new = np.clip(P_new, 0, None)
        P_new = P_new / np.sum(P_new, axis=1, keepdims=True)  # Normalize

        current_violation = chance_violation_prob(P_new)
        if current_violation > best_violation:
            best_violation = current_violation
            best_P_k = P_new.copy()

        if best_violation >= (1 - eta):
            break

    constraint_satisfied = best_violation <= (1 - eta)
    return constraint_satisfied, best_P_k

# ✅ Calculate and print Wasserstein distance between initial and optimized distributions
def print_final_wasserstein_distance(P_0, best_P_k):
    distance = wasserstein_distance(P_0, best_P_k)
    print(f"\n📏 Final Wasserstein Distance between Initial (P_0) and Best Distribution (P_k): {distance:.4f}")


def calculate_demand_satisfaction(Y, S, X, P):
    S_expanded = np.expand_dims(S, axis=1)  # Shape (J, 1, T) to match (J, I, T)
    Y = np.minimum(Y, S_expanded) * X[:, np.newaxis, :]  # Ensure allocations are feasible
    # Count how many scenarios violate the constraint (Y > S)
    violating_scenarios = np.sum([
        np.any(Y[:, :, t] > S_expanded[:, :, t]) for t in range(S.shape[1])
    ])
    total_scenarios = P.shape[0]  # Number of demand scenarios
    demand_satisfaction = (1 - violating_scenarios / total_scenarios) * 100  # Convert to percentage

    return demand_satisfaction

# Step 7: Iterative Optimization Framework

def optimize_facility_location(I, J, T, f, a, c, d, rho, P_0, eps, eta, q_t, max_iter=15):
    start_time = time.time()

    P_k = P_0
    X, S, Y = generate_initial_solution_ga(I, J, T, q_t, P_0)
    # Print outputs of GA
    print("\n✅ Generated Initial Solution using Genetic Algorithm:\n")
    print(f"X (Facility Opening Decisions):\n{X}\n")
    #print(f"S (Storage Levels):\n{S}\n")
    #print(f"Y (Allocations to Customers):\n{Y}\n")
    best_X, best_S, best_Y, best_P_k = X, S, Y, P_k
    best_obj_val = float('inf')
    second_stage_costs = []  # Track second-stage costs over iterations
    Objective_Value = []  # Track second-stage costs over iterations


    for k in range(max_iter):
        # Get the best P_k that minimizes second-stage cost
        best_P_k = solve_cjko(P_k, eps, c, d, rho, Y, S)
        # Verify chance constraints
        constraint_satisfied, best_P_k = check_chance_constraints_cjko(Y, S, eta, X, best_P_k, P_0, eps)

        if constraint_satisfied:
            # First-stage cost calculation
            Z = np.zeros_like(X)
            Z[:, 0] = X[:, 0]
            Z[:, 1:] = np.logical_and(X[:, 1:], np.logical_not(X[:, :-1])).astype(int)
            term1 = np.sum(f * Z + a * (S - np.roll(S, shift=1, axis=1)))
            # Updated expected cost calculation using best_P_k
            expected_cost = np.mean([
                np.sum(c * (Y * best_P_k[scenario][:, np.newaxis])) +
                np.sum(rho[np.newaxis, :, np.newaxis] *
                       np.maximum(0, Y * best_P_k[scenario][:, np.newaxis] - S[:, np.newaxis, :]))
                for scenario in range(best_P_k.shape[0])
            ])

            facility_open_violation = np.sum(np.all(X == 0, axis=0)) * 1e6


            obj_val = term1 + expected_cost+ facility_open_violation
            second_stage_costs.append(expected_cost)  # Store cost for plotting
            Objective_Value.append(obj_val)

            print(
                f"Iteration {k}: First-Stage Cost = {term1:.4f}, Second-Stage Expected Cost = {expected_cost:.4f}, "
                f"Total Objective Value = {obj_val:.4f}, Epsilon = {eps:.4f}"
            )

            # Update the best found solution
            if obj_val < best_obj_val:
                best_obj_val = obj_val
                best_X, best_S, best_Y, best_P_k = X, S, Y, best_P_k

            # Adjust storage and allocations
            S = np.maximum(S + np.random.randint(-2, 3, size=S.shape), 0) * X
            Y = np.minimum(Y, S[:, np.newaxis, :])
        else:
            print(f"Iteration {k}: Constraint Violated, Adjusting Epsilon to {eps}")
            # Verify chance constraints and compute demand satisfaction
        constraint_satisfied, best_P_k = check_chance_constraints_cjko(Y, S, eta, X, best_P_k, P_0, eps)
        demand_satisfaction = calculate_demand_satisfaction(Y, S, X, best_P_k)

        print(f"Iteration {k}: Demand Satisfaction = {demand_satisfaction:.2f}%")

    end_time = time.time()
    print(f"Facility Location Iteration: {end_time - start_time} seconds")
    # Plot second-stage expected cost over iterations

    print("\n✅ Best Optimized Solution:")
    print(f"Best Objective Value: {best_obj_val:.4f}")
    print(f"X (Facility Decisions):\n{best_X}")
    print(f"S (Storage Levels):\n{best_S}")
    print(f"Y (Allocations):\n{best_Y}")

    plt.figure(figsize=(6, 3))
    sns.heatmap(X, annot=True, cmap="Blues", cbar=False, xticklabels=[f"T{t}" for t in range(X.shape[1])],
                yticklabels=[f"Facility {j}" for j in range(X.shape[0])])
    plt.title("Facility Opening Decisions (X)")
    plt.xlabel("Time Period")
    plt.ylabel("Facility")
    plt.show()
    plt.figure(figsize=(6, 3))
    sns.heatmap(S, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=[f"T{t}" for t in range(S.shape[1])],
                yticklabels=[f"Facility {j}" for j in range(S.shape[0])])
    plt.title("Storage Levels (S)")
    plt.xlabel("Time Period")
    plt.ylabel("Facility")
    plt.show()
    I, J, T = Y.shape[1], Y.shape[0], Y.shape[2]

    for j in range(J):
        plt.figure(figsize=(8, 3))
        for i in range(I):
            plt.bar(range(T), Y[j, i], label=f"Customer {i}", bottom=np.sum(Y[j, :i], axis=0))
        plt.title(f"Allocations from Facility {j}")
        plt.xlabel("Time Period")
        plt.ylabel("Allocated Quantity")
        plt.xticks(range(T), [f"T{t}" for t in range(T)])
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(range(len(Objective_Value)), Objective_Value, marker='o', linestyle='-', color='b',
                 label="Objective_Value")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.title("Convergence of Objective Value")
        plt.legend()
        plt.grid(True)
        plt.show()
    return best_X, best_S, best_Y, best_P_k
