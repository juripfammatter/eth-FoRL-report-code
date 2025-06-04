import os

from tqdm import tqdm

from congestion_games import *
import matplotlib.pyplot as plt
import itertools
import numpy as np
import copy
import statistics
import seaborn as sns
import time
from concurrent.futures import ProcessPoolExecutor


def projection_simplex_sort(v, z=1):
    # Courtesy: EdwardRaff/projection_simplex.py
    if v.sum() == z and np.alltrue(v >= 0):
        return v
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


# Define the states and some necessary info
N = 8  # number of agents
harm = -100 * N  # pentalty for being in bad state

safe_state = CongGame(N, 1, [[1, 0], [2, 0], [4, 0], [6, 0]])
bad_state = CongGame(N, 1, [[1, -100], [2, -100], [4, -100], [6, -100]])
state_dic = {0: safe_state, 1: bad_state}

M = safe_state.num_actions
D = safe_state.m  # number facilities
S = 2

# Dictionary to store the action profiles and rewards to
selected_profiles = {}

# Dictionary associating each action (value) to an integer (key)
act_dic = {}
counter = 0
for act in safe_state.actions:
    act_dic[counter] = act
    counter += 1


def get_next_state(state, actions):
    acts_from_ints = [act_dic[i] for i in actions]
    density = state_dic[state].get_counts(acts_from_ints)
    max_density = max(density)

    if state == 0 and max_density > N / 2 or state == 1 and max_density > N / 4:
        # if state == 0 and max_density > N/2 and np.random.uniform() > 0.2 or state == 1 and max_density > N/4 and np.random.uniform() > 0.1:
        return 1
    return 0


def pick_action(prob_dist):
    # np.random.choice(range(len(prob_dist)), 1, p = prob_dist)[0]
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p=prob_dist)
    return action[0]


def visit_dist(state, policy, gamma, T, samples):
    # This is the unnormalized visitation distribution. Since we take finite trajectories, the normalization constant is (1-gamma**T)/(1-gamma).
    visit_states = {st: np.zeros(T) for st in range(S)}
    for i in range(samples):
        curr_state = state
        for t in range(T):
            visit_states[curr_state][t] += 1
            actions = [pick_action(policy[curr_state, i]) for i in range(N)]
            curr_state = get_next_state(curr_state, actions)
    dist = [
        np.dot(v / samples, gamma ** np.arange(T)) for (k, v) in visit_states.items()
    ]
    return dist


def value_function(policy, gamma, T, samples):
    value_fun = {(s, i): 0 for s in range(S) for i in range(N)}
    for k in range(samples):
        for state in range(S):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]
                q = tuple(actions + [curr_state])
                rewards = selected_profiles.setdefault(
                    q, get_reward(state_dic[curr_state], [act_dic[i] for i in actions])
                )
                for i in range(N):
                    value_fun[state, i] += (gamma**t) * rewards[i]
                curr_state = get_next_state(curr_state, actions)
    value_fun.update((x, v / samples) for (x, v) in value_fun.items())
    return value_fun


def Q_function(agent, state, action, policy, gamma, value_fun, samples):
    """TD(0) estimate of the Q-function for a given agent, state and action."""

    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        actions[agent] = action
        q = tuple(actions + [state])
        rewards = selected_profiles.setdefault(
            q, get_reward(state_dic[state], [act_dic[i] for i in actions])
        )
        tot_reward += (
            rewards[agent] + gamma * value_fun[get_next_state(state, actions), agent]
        )
    return tot_reward / samples


def policy_accuracy(policy_pi, policy_star):
    total_dif = N * [0]
    for agent in range(N):
        for state in range(S):
            total_dif[agent] += np.sum(
                np.abs((policy_pi[state, agent] - policy_star[state, agent]))
            )
    # total_dif[agent] += np.sqrt(np.sum((policy_pi[state, agent] - policy_star[state, agent])**2))
    return np.sum(total_dif) / N


def policy_gradient(mu, max_iters, gamma, eta, T, samples):

    policy = {(s, i): [1 / M] * M for s in range(S) for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]

    for t in tqdm(range(max_iters)):

        # print(t)

        b_dist = M * [0]
        for st in range(S):
            a_dist = visit_dist(st, policy, gamma, T, samples)

            b_dist[st] = np.dot(a_dist, mu)

        grads = np.zeros((N, S, M))
        value_fun = value_function(policy, gamma, T, samples)

        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    grads[agent, st, act] = b_dist[st] * Q_function(
                        agent, st, act, policy, gamma, value_fun, samples
                    )

        for agent in range(N):
            for st in range(S):
                policy[st, agent] = projection_simplex_sort(
                    np.add(policy[st, agent], eta[agent] * grads[agent, st]), z=1
                )
        policy_hist.append(copy.deepcopy(policy))

        if policy_accuracy(policy_hist[t], policy_hist[t - 1]) < 10e-16:
            # if policy_accuracy(policy_hist[t+1], policy_hist[t]) < 10e-16: (it makes a difference, not when t=0 but from t=1 onwards.)
            return policy_hist

    return policy_hist

# Our contribution
def softmax(theta):
    num = np.exp(theta)
    denom = np.sum(num)
    return num / denom

# Our contribution
def policy_gradient_softmax(mu, max_iters, gamma, eta, T, samples):

    policy = {(s, i): [1 / M] * M for s in range(S) for i in range(N)}
    theta = {(s, i): [1 / M] * M for s in range(S) for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]

    for t in tqdm(range(max_iters)):

        b_dist = M * [0]
        for st in range(S):
            a_dist = visit_dist(st, policy, gamma, T, samples)

            b_dist[st] = np.dot(a_dist, mu)

        Q = np.zeros((N, S, M))
        grads_theta = np.zeros((N, S, M))
        value_fun = value_function(policy, gamma, T, samples)

        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    Q[agent, st, act] = Q_function(
                        agent, st, act, policy, gamma, value_fun, samples
                    )
                    grads_theta[agent, st] = b_dist[st] * (
                        Q[agent, st] - np.dot(Q[agent, st], policy[st, agent])
                    )

        for agent in range(N):
            for st in range(S):
                theta[st, agent] = np.add(
                    theta[st, agent],
                    eta[agent] / (1 - gamma) * grads_theta[agent, st],
                )
                policy[st, agent] = softmax(theta[st, agent])
        policy_hist.append(copy.deepcopy(policy))

        if policy_accuracy(policy_hist[t], policy_hist[t - 1]) < 10e-16:
            # if policy_accuracy(policy_hist[t+1], policy_hist[t]) < 10e-16: (it makes a difference, not when t=0 but from t=1 onwards.)
            return policy_hist

    return policy_hist


def get_accuracies(policy_hist):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(policy_hist[i], fin)
        accuracies.append(this_acc)
    return accuracies

# Our contribution
def process_run(k, iters, eta, T, samples, param: str):
    if param == "direct":
        policy_hist = policy_gradient([0.5, 0.5], iters, 0.99, eta, T, samples)
    elif param == "softmax":
        policy_hist = policy_gradient_softmax([0.5, 0.5], iters, 0.99, eta, T, samples)
    else:
        raise ValueError("Unknown parametrization: {}".format(param))
    raw_accuracy = get_accuracies(policy_hist)
    converged_policy = policy_hist[-1]
    density = np.zeros((S, M))
    for i in range(N):
        for s in range(S):
            density[s] += converged_policy[s, i]
    return raw_accuracy, density

# Our contribution
def run_experiment(runs, iters, eta, T, samples, param="direct"):
    parallel = False # run sequentially for reproducibility
    densities = np.zeros((S, M))
    raw_accuracies = []

    if not parallel:
        for k in range(runs):
            if param == "direct":
                policy_hist = policy_gradient([0.5, 0.5], iters, 0.99, eta, T, samples)
            elif param == "softmax":
                policy_hist = policy_gradient_softmax([0.5, 0.5], iters, 0.99, eta, T, samples)
            else:
                raise ValueError("Unknown parametrization: {}".format(param))
            raw_accuracies.append(get_accuracies(policy_hist))

            converged_policy = policy_hist[-1]
            for i in range(N):
                for s in range(S):
                    densities[s] += converged_policy[s, i]

        densities = densities / runs

    else:
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    process_run,
                    range(runs),
                    [iters] * runs,
                    [eta] * runs,
                    [T] * runs,
                    [samples] * runs,
                    [param] * runs,
                )
            )

        for raw_accuracy, density in results:
            raw_accuracies.append(raw_accuracy)
            densities += density

        densities = densities / runs

    return raw_accuracies, densities

# Our contributions: modifications to the original code to allow for parallel execution, and comparison of different parametrizations
def full_experiment(runs, iters, T, samples):
    eta_direct = 1e-4 * np.ones((N, 1))
    raw_accuracies_direct, densities_direct = run_experiment(
        runs, iters, eta_direct, T, samples, param="direct"
    )

    eta_softmax = 1e-5 * np.ones((N, 1))
    raw_accuracies_softmax, densities_softmax = run_experiment(
        runs, iters, eta_softmax, T, samples, param="softmax"
    )

    plot_accuracies_direct = np.array(
        list(itertools.zip_longest(*raw_accuracies_direct, fillvalue=np.nan))
    ).T

    plot_accuracies_softmax = np.array(
        list(itertools.zip_longest(*raw_accuracies_softmax, fillvalue=np.nan))
    ).T

    clrs = sns.color_palette("viridis", 3)
    piters = list(range(plot_accuracies_direct.shape[1]))

    sns.set(font_scale=1.35, style="darkgrid")

    fig2 = plt.figure(figsize=(6, 5), dpi=300)
    for i in range(len(plot_accuracies_direct)):
        plt.plot(piters, plot_accuracies_direct[i])
    plt.grid(linewidth=0.6)
    plt.gca().set(
        xlabel="Iterations",
        ylabel="L1-accuracy",
        title="Direct Parametrization:\n agents = {}, runs = {}".format(N, runs),
    )
    plt.show()
    os.makedirs("plots", exist_ok=True)
    fig2.savefig("plots/individual_runs_n{}.png".format(N), bbox_inches="tight")

    fig1, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)

    for i, (acc, name) in enumerate(
        zip([plot_accuracies_direct, plot_accuracies_softmax], ["Direct", "Softmax"])
    ):
        piters = list(range(acc.shape[1]))
        acc = np.nan_to_num(acc)
        pmean = list(map(statistics.mean, zip(*acc)))
        pstdv = list(map(statistics.stdev, zip(*acc)))

        ax.plot(piters, pmean, color=clrs[i], label=f"Mean L1-accuracy ({name})")
        ax.fill_between(
            piters,
            np.clip(np.subtract(pmean, pstdv), a_min=0, a_max=None),
            np.add(pmean, pstdv),
            alpha=0.3,
            facecolor=clrs[i],
            label="1-standard deviation",
        )
    ax.legend(facecolor="white")
    plt.grid(linewidth=0.6)
    ax.set(
        xlabel="Iterations",
        ylabel="L1-accuracy",
        title="Direct vs. softmax parametrization:\n agents = {}, runs = {}".format(
            N, runs
        ),
    )
    plt.tight_layout()
    plt.show()
    fig1.savefig("plots/avg_runs_n{}.png".format(N), bbox_inches="tight")

    fig3, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    index = np.arange(D)
    bar_width = 0.35
    opacity = 1

    for ax, den, name in zip(
        ax, [densities_direct, densities_softmax], ["Direct", "Softmax"]
    ):
        ax.bar(
            index,
            den[0],
            bar_width,
            alpha=0.7 * opacity,
            color="b",
            label="Safe state",
        )

        ax.bar(
            index + bar_width,
            den[1],
            bar_width,
            alpha=opacity,
            color="r",
            label="Distancing state",
        )

        ax.set(
            xlabel="Facility",
            ylabel="Average number of agents",
            title="{} parametrization:\n agents = {}, runs = {}".format(name, N, runs),
        )
        ax.set_xticks(index + bar_width / 2, ("A", "B", "C", "D"))
        ax.legend(facecolor="white")
    plt.tight_layout()
    fig3.savefig("plots/facilities_n{}.png".format(N))
    plt.show()

    return fig1, fig2, fig3


def main():
    np.random.seed(66)  # For reproducibility
    start_time = time.time()
    full_experiment(8, 500, 20, 10)

    elapsed_time = time.time() - start_time
    print(elapsed_time)


if __name__ == "__main__":
    main()
