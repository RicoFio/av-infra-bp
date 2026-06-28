from itertools import product
import gurobipy as gp
from gurobipy import GRB

def compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for a in range(n + 1):
        for tail in compositions(n - a, k - 1):
            yield (a,) + tail


def solve_nonatomic_disclosure_lp(
    *,
    edges,
    states,
    prior,
    types,
    demand,
    routes,
    alpha,
    beta,
    menus,
    tol=1e-9,
    verbose=False,
):
    """
    Nonatomic typed lottery LP over a finite route-flow support.

    routes[t][r]      = list of edges on route r for type t
    demand[t]         = total number of agents of type t (positive integer)
    alpha[(e, th)]    = affine slope of edge e in state th
    beta[(e, th)]     = affine intercept of edge e in state th
    menus[(t,r)][j]   = disclosed edge-cost vector d, with d[e] for all e
    """

    E = list(edges)
    TH = list(states)
    T = list(types)

    TR = [(t, r) for t in T for r in range(len(routes[t]))]
    tr_index = {tr: i for i, tr in enumerate(TR)}
    route_edges = {(t, r): set(routes[t][r]) for (t, r) in TR}

    # Hypergraph factorization: edge e depends only on routes using e.
    edge_users = {
        e: [(t, r) for (t, r) in TR if e in route_edges[(t, r)]]
        for e in E
    }

    S_e = {e: sorted({t for (t, _) in edge_users[e]}) for e in E}
    unary_edges = [e for e in E if len(S_e[e]) <= 1]
    coupling_edges = [e for e in E if len(S_e[e]) >= 2]

    # Check that each disclosure vector induces its intended route.
    for (t, r) in TR:
        if (t, r) not in menus or len(menus[(t, r)]) == 0:
            raise ValueError(f"missing disclosure menu for {(t, r)}")

        for j, d in enumerate(menus[(t, r)]):
            missing = [e for e in E if e not in d]
            if missing:
                raise ValueError(f"menu {(t, r)}[{j}] misses edges {missing}")

            lhs = sum(d[e] for e in route_edges[(t, r)])
            for q in range(len(routes[t])):
                rhs = sum(d[e] for e in route_edges[(t, q)])
                if lhs > rhs + tol:
                    raise ValueError(
                        f"menu {(t, r)}[{j}] does not induce route {r}"
                    )

    # Enumerate integer route-flow splits: each agent is 1 unit.
    per_type_splits = {}
    for t in T:
        N = int(demand[t])
        if N <= 0:
            raise ValueError(f"demand[{t}] must be a positive integer")

        k = len(routes[t])
        per_type_splits[t] = [
            tuple(c for c in comp)
            for comp in compositions(N, k)
        ]

    profiles = []
    for split_tuple in product(*(per_type_splits[t] for t in T)):
        x = [0.0] * len(TR)
        for t, split in zip(T, split_tuple):
            for r, flow in enumerate(split):
                x[tr_index[(t, r)]] = flow
        profiles.append(tuple(x))

    # Loads use the factorization edge_users[e].
    loads = []
    for x in profiles:
        loads.append({
            e: sum(x[tr_index[(t, r)]] for (t, r) in edge_users[e])
            for e in E
        })

    # Real edge costs.
    real_c = {}
    for th in TH:
        for p, load in enumerate(loads):
            for e in E:
                real_c[(th, p, e)] = alpha[(e, th)] * load[e] + beta[(e, th)]

    # Sender objective: total travel time.
    F = {}
    for th in TH:
        for p, load in enumerate(loads):
            F[(th, p)] = sum(load[e] * real_c[(th, p, e)] for e in E)

    # Nonatomic route costs: deviation does not alter load.
    route_cost = {}
    delta = {}
    for th in TH:
        for p in range(len(profiles)):
            for (t, r) in TR:
                route_cost[(th, p, t, r)] = sum(
                    real_c[(th, p, e)] for e in route_edges[(t, r)]
                )

            for t in T:
                for r in range(len(routes[t])):
                    for q in range(len(routes[t])):
                        delta[(th, p, t, r, q)] = (
                            route_cost[(th, p, t, r)]
                            - route_cost[(th, p, t, q)]
                        )

    m = gp.Model("nonatomic_edge_disclosure_lottery_lp")
    m.Params.OutputFlag = 1 if verbose else 0

    phi = {}
    by_state = {th: [] for th in TH}

    # Variables phi[theta, profile, gamma].
    for th in TH:
        for p, x in enumerate(profiles):
            choices = []
            for i, tr in enumerate(TR):
                choices.append([0] if x[i] <= tol else range(len(menus[tr])))

            for gamma in product(*choices):
                feasible = True

                for i, tr in enumerate(TR):
                    if x[i] <= tol:
                        continue

                    d = menus[tr][gamma[i]]

                    # Main disclosure feasibility:
                    # disclosed_c >= real_c edgewise.
                    if any(d[e] + tol < real_c[(th, p, e)] for e in E):
                        feasible = False
                        break

                if feasible:
                    v = m.addVar(lb=0.0, name=f"phi[{th},{p},{gamma}]")
                    phi[(th, p, gamma)] = v
                    by_state[th].append(v)

    # Statewise probability normalization.
    for th in TH:
        if not by_state[th]:
            raise ValueError(f"no feasible outcome for state {th}")
        m.addConstr(gp.quicksum(by_state[th]) == 1.0, name=f"prob[{th}]")

    # Lottery obedience:
    #
    # For each type t, disclosure-induced route r, deviation q, and observed disclosed
    # edge-cost vector d_id, require:
    #
    # E[ mass receiving d_id on r * (cost(r) - cost(q)) ] <= 0.
    #
    # Nonatomic: cost(q) is evaluated at the same load profile.
    for t in T:
        for r in range(len(routes[t])):
            i = tr_index[(t, r)]

            for q in range(len(routes[t])):
                if q == r:
                    continue

                for d_id in range(len(menus[(t, r)])):
                    expr = gp.LinExpr()

                    for (th, p, gamma), v in phi.items():
                        x_tr = profiles[p][i]

                        if x_tr > tol and gamma[i] == d_id:
                            expr += (
                                prior[th]
                                * x_tr
                                * delta[(th, p, t, r, q)]
                                * v
                            )

                    m.addConstr(expr <= 0.0, name=f"obed[{t},{r}->{q},d={d_id}]")

    m.setObjective(
        gp.quicksum(
            prior[th] * F[(th, p)] * v
            for (th, p, gamma), v in phi.items()
        ),
        GRB.MINIMIZE,
    )
    
    # For debugging
    m.update()
    print("NumBinVars", m.NumBinVars)
    print("NumConstrs", m.NumConstrs)
    print("NumIntVars", m.NumIntVars)
    print("NumVars", m.NumVars)

    m.optimize()

    result = {
        "model": m,
        "TR": TR,
        "profiles": profiles,
        "loads": loads,
        "S_e": S_e,
        "unary_edges": unary_edges,
        "coupling_edges": coupling_edges,
        "policy": [],
    }

    if m.Status == GRB.OPTIMAL:
        result["objective"] = m.ObjVal

        for (th, p, gamma), v in phi.items():
            if v.X > 1e-8:
                x = profiles[p]
                result["policy"].append({
                    "state": th,
                    "prob_given_state": v.X,
                    "profile": {
                        tr: x[tr_index[tr]]
                        for tr in TR
                    },
                    "disclosure_ids": {
                        tr: gamma[tr_index[tr]]
                        for tr in TR
                        if x[tr_index[tr]] > tol
                    },
                })

    return result


def toy_example():
    edges = ["a", "b", "c"]
    states = ["normal", "incident"]
    prior = {"normal": 0.7, "incident": 0.3}

    types = ["OD1", "OD2"]
    demand = {"OD1": 2, "OD2": 2}  # 2 people on each OD pair

    routes = {
        "OD1": [["a"], ["b"]],
        "OD2": [["b"], ["c"]],
    }

    alpha, beta = {}, {}
    for e in edges:
        for th in states:
            alpha[(e, th)] = 0.2
            beta[(e, th)] = 1.0

    beta[("b", "incident")] = 2.5

    # Each disclosed vector makes its intended route shortest.
    # Also, feasibility later requires disclosed_c >= real_c.
    menus = {
        ("OD1", 0): [{"a": 5.0, "b": 6.0, "c": 9.0}],
        ("OD1", 1): [{"a": 6.0, "b": 5.0, "c": 9.0}],
        ("OD2", 0): [{"a": 9.0, "b": 5.0, "c": 6.0}],
        ("OD2", 1): [{"a": 9.0, "b": 6.0, "c": 5.0}],
    }

    res = solve_nonatomic_disclosure_lp(
        edges=edges,
        states=states,
        prior=prior,
        types=types,
        demand=demand,
        routes=routes,
        alpha=alpha,
        beta=beta,
        menus=menus,
        verbose=False,
    )

    print("objective:", res.get("objective"))
    print("unary edges:", res["unary_edges"])
    print("coupling edges:", res["coupling_edges"])

    for row in res["policy"]:
        print(row)


if __name__=="__main__":
    toy_example()
