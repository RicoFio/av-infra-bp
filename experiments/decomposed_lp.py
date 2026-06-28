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


def _type_components(T, S_e):
    """
    Partition types into connected components based on shared coupling edges.

    Two types are adjacent if they both appear in S_e[e] for some edge e.
    Returns a list of frozensets, one per component.
    """
    parent = {t: t for t in T}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for types_on_edge in S_e.values():
        for i in range(1, len(types_on_edge)):
            union(types_on_edge[0], types_on_edge[i])

    groups = {}
    for t in T:
        root = find(t)
        groups.setdefault(root, []).append(t)

    return [frozenset(g) for g in groups.values()]


def _solve_component_lp(
    *,
    component,
    edges,
    states,
    prior,
    demand,
    routes,
    alpha,
    beta,
    menus,
    grid_units,
    per_type_splits,
    route_edges,
    tol,
    verbose,
):
    """
    Build and solve the disclosure LP for a single connected component.

    Returns a dict with keys: model, phi, TR, tr_index, profiles, loads,
    objective (if optimal), policy.
    """
    T_C = sorted(component, key=lambda t: list(edges).index(list(routes[t][0])[0])
                 if routes[t] else t)
    # Preserve original type ordering within the component
    T_C = [t for t in list(routes.keys()) if t in component]

    TR_C = [(t, r) for t in T_C for r in range(len(routes[t]))]
    tr_index_C = {tr: i for i, tr in enumerate(TR_C)}
    route_edges_C = {(t, r): route_edges[(t, r)] for (t, r) in TR_C}

    E_C = sorted({e for tr in TR_C for e in route_edges_C[tr]},
                 key=list(edges).index)

    edge_users_C = {
        e: [(t, r) for (t, r) in TR_C if e in route_edges_C[(t, r)]]
        for e in E_C
    }

    profiles_C = []
    for split_tuple in product(*(per_type_splits[t] for t in T_C)):
        x = [0.0] * len(TR_C)
        for t, split in zip(T_C, split_tuple):
            for r, flow in enumerate(split):
                x[tr_index_C[(t, r)]] = flow
        profiles_C.append(tuple(x))

    loads_C = []
    for x in profiles_C:
        loads_C.append({
            e: sum(x[tr_index_C[(t, r)]] for (t, r) in edge_users_C[e])
            for e in E_C
        })

    real_c_C = {}
    for th in states:
        for p, load in enumerate(loads_C):
            for e in E_C:
                real_c_C[(th, p, e)] = alpha[(e, th)] * load[e] + beta[(e, th)]

    F_C = {}
    for th in states:
        for p, load in enumerate(loads_C):
            F_C[(th, p)] = sum(load[e] * real_c_C[(th, p, e)] for e in E_C)

    route_cost_C = {}
    delta_C = {}
    for th in states:
        for p in range(len(profiles_C)):
            for (t, r) in TR_C:
                route_cost_C[(th, p, t, r)] = sum(
                    real_c_C[(th, p, e)] for e in route_edges_C[(t, r)]
                )
            for t in T_C:
                for r in range(len(routes[t])):
                    for q in range(len(routes[t])):
                        delta_C[(th, p, t, r, q)] = (
                            route_cost_C[(th, p, t, r)]
                            - route_cost_C[(th, p, t, q)]
                        )

    m = gp.Model(f"component_lp_{'_'.join(str(t) for t in T_C)}")
    m.Params.OutputFlag = 1 if verbose else 0

    phi_C = {}
    by_state_C = {th: [] for th in states}

    for th in states:
        for p, x in enumerate(profiles_C):
            choices = []
            for i, tr in enumerate(TR_C):
                choices.append([0] if x[i] <= tol else range(len(menus[tr])))

            for gamma in product(*choices):
                feasible = True
                for i, tr in enumerate(TR_C):
                    if x[i] <= tol:
                        continue
                    d = menus[tr][gamma[i]]
                    if any(d[e] + tol < real_c_C[(th, p, e)] for e in E_C):
                        feasible = False
                        break
                if feasible:
                    v = m.addVar(lb=0.0, name=f"phi[{th},{p},{gamma}]")
                    phi_C[(th, p, gamma)] = v
                    by_state_C[th].append(v)

    for th in states:
        if not by_state_C[th]:
            raise ValueError(f"no feasible outcome for state {th} in component {T_C}")
        m.addConstr(gp.quicksum(by_state_C[th]) == 1.0, name=f"prob[{th}]")

    for t in T_C:
        for r in range(len(routes[t])):
            i = tr_index_C[(t, r)]
            for q in range(len(routes[t])):
                if q == r:
                    continue
                for d_id in range(len(menus[(t, r)])):
                    expr = gp.LinExpr()
                    for (th, p, gamma), v in phi_C.items():
                        x_tr = profiles_C[p][i]
                        if x_tr > tol and gamma[i] == d_id:
                            expr += (
                                prior[th]
                                * x_tr
                                * delta_C[(th, p, t, r, q)]
                                * v
                            )
                    m.addConstr(expr <= 0.0, name=f"obed[{t},{r}->{q},d={d_id}]")

    m.setObjective(
        gp.quicksum(
            prior[th] * F_C[(th, p)] * v
            for (th, p, gamma), v in phi_C.items()
        ),
        GRB.MINIMIZE,
    )

    m.update()
    print(
        f"Component {T_C}: "
        f"NumVars={m.NumVars}, NumConstrs={m.NumConstrs}"
    )
    m.optimize()

    result = {
        "model": m,
        "phi": phi_C,
        "TR": TR_C,
        "tr_index": tr_index_C,
        "profiles": profiles_C,
        "loads": loads_C,
        "policy": [],
    }

    if m.Status == GRB.OPTIMAL:
        result["objective"] = m.ObjVal
        for (th, p, gamma), v in phi_C.items():
            if v.X > 1e-8:
                x = profiles_C[p]
                result["policy"].append({
                    "state": th,
                    "prob_given_state": v.X,
                    "profile": {tr: x[tr_index_C[tr]] for tr in TR_C},
                    "disclosure_ids": {
                        tr: gamma[tr_index_C[tr]]
                        for tr in TR_C
                        if x[tr_index_C[tr]] > tol
                    },
                })

    return result


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
    grid_units,
    tol=1e-9,
    verbose=False,
):
    """
    Nonatomic typed lottery LP over a finite route-flow support.

    routes[t][r]      = list of edges on route r for type t
    demand[t]         = nonatomic mass of type t
    grid_units[t]     = number of grid quanta for type-t split
    alpha[(e, th)]    = affine slope of edge e in state th
    beta[(e, th)]     = affine intercept of edge e in state th
    menus[(t,r)][j]   = disclosed edge-cost vector d, with d[e] for all e

    When types are connected only through shared edges, the LP decomposes into
    independent sub-LPs — one per connected component of the type-interaction
    graph. Each component is solved separately; objectives are summed and
    policies are combined by Cartesian product.
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

    # Enumerate finite nonatomic route-flow splits (shared across components).
    per_type_splits = {}
    for t in T:
        U = int(grid_units[t])
        if U <= 0:
            raise ValueError(f"grid_units[{t}] must be positive")
        k = len(routes[t])
        per_type_splits[t] = [
            tuple(demand[t] * c / U for c in comp)
            for comp in compositions(U, k)
        ]

    # Detect connected components of the type-interaction graph.
    components = _type_components(T, S_e)
    print(f"Components: {[sorted(c) for c in components]}")

    # --- Single-component fast path (original code, no overhead) ---
    if len(components) == 1:
        profiles = []
        for split_tuple in product(*(per_type_splits[t] for t in T)):
            x = [0.0] * len(TR)
            for t, split in zip(T, split_tuple):
                for r, flow in enumerate(split):
                    x[tr_index[(t, r)]] = flow
            profiles.append(tuple(x))

        loads = []
        for x in profiles:
            loads.append({
                e: sum(x[tr_index[(t, r)]] for (t, r) in edge_users[e])
                for e in E
            })

        real_c = {}
        for th in TH:
            for p, load in enumerate(loads):
                for e in E:
                    real_c[(th, p, e)] = alpha[(e, th)] * load[e] + beta[(e, th)]

        F = {}
        for th in TH:
            for p, load in enumerate(loads):
                F[(th, p)] = sum(load[e] * real_c[(th, p, e)] for e in E)

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
                        if any(d[e] + tol < real_c[(th, p, e)] for e in E):
                            feasible = False
                            break
                    if feasible:
                        v = m.addVar(lb=0.0, name=f"phi[{th},{p},{gamma}]")
                        phi[(th, p, gamma)] = v
                        by_state[th].append(v)

        for th in TH:
            if not by_state[th]:
                raise ValueError(f"no feasible outcome for state {th}")
            m.addConstr(gp.quicksum(by_state[th]) == 1.0, name=f"prob[{th}]")

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
            "components": components,
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
                        "profile": {tr: x[tr_index[tr]] for tr in TR},
                        "disclosure_ids": {
                            tr: gamma[tr_index[tr]]
                            for tr in TR
                            if x[tr_index[tr]] > tol
                        },
                    })

        return result

    # --- Multi-component decomposition path ---
    comp_results = []
    for comp in components:
        cr = _solve_component_lp(
            component=comp,
            edges=E,
            states=TH,
            prior=prior,
            demand=demand,
            routes=routes,
            alpha=alpha,
            beta=beta,
            menus=menus,
            grid_units=grid_units,
            per_type_splits=per_type_splits,
            route_edges=route_edges,
            tol=tol,
            verbose=verbose,
        )
        comp_results.append(cr)

    # Check all components solved optimally.
    for cr in comp_results:
        if cr["model"].Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Component LP did not solve to optimality "
                f"(status {cr['model'].Status})"
            )

    total_objective = sum(cr["objective"] for cr in comp_results)

    # Merge policies: Cartesian product across components, per state.
    merged_policy = []
    for th in TH:
        per_comp = [
            [e for e in cr["policy"] if e["state"] == th]
            for cr in comp_results
        ]
        for joint in product(*per_comp):
            prob = 1.0
            profile = {}
            disc = {}
            for entry in joint:
                prob *= entry["prob_given_state"]
                profile.update(entry["profile"])
                disc.update(entry["disclosure_ids"])
            merged_policy.append({
                "state": th,
                "prob_given_state": prob,
                "profile": profile,
                "disclosure_ids": disc,
            })

    return {
        "model": [cr["model"] for cr in comp_results],
        "TR": TR,
        "profiles": None,   # not meaningful for multi-component
        "loads": None,
        "S_e": S_e,
        "unary_edges": unary_edges,
        "coupling_edges": coupling_edges,
        "components": components,
        "objective": total_objective,
        "policy": merged_policy,
    }
