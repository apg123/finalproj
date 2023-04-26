import codecs
from collections import defaultdict
import math
import numpy as np
#import matplotlib.pyplot as plt
from pathlib import Path
import csv
import cvxpy as cp
import sklearn as sk
from sklearn.cluster import AgglomerativeClustering



# Classes from Pabulib
class Voter:
    def __init__(self,
            id : str,
            sex : str = None,
            age : int = None,
            subunits : set[str] = set(),
            utilities : list = list()
            ):
        self.id = id #unique id
        self.sex = sex
        self.age = age
        self.subunits = subunits
        self.utilities = list()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, v):
        return self.id == v.id

    def __repr__(self):
        return f"v({self.id})"

class Candidate:
    def __init__(self,
            id : str,
            cost : int,
            name : str = None,
            subunit : str = None
            ):
        self.id = id #unique id
        self.cost = cost
        self.name = name
        self.subunit = subunit #None for citywide projects

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, c):
        return self.id == c.id

    def __repr__(self):
        return f"c({self.id})"

class Election:
    def __init__(self,
            name : str = None,
            voters : set[Voter] = None,
            profile : dict[Candidate, dict[Voter, int]] = None,
            budget : int = 0,
            subunits : set[str] = None
            ):
        self.name = name
        self.voters = voters if voters else set()
        self.profile = profile if profile else {} #dict: candidates -> voters -> score
        self.budget = budget
        self.subunits = subunits if subunits else set()

    def binary_to_cost_utilities(self):
        assert all((self.profile[c][v] == 1) for c in self.profile for v in self.profile[c])
        return self.score_to_cost_utilities()

    def cost_to_binary_utilities(self):
        assert all((self.profile[c][v] == c.cost) for c in self.profile for v in self.profile[c])
        return self.cost_to_score_utilities()

    def score_to_cost_utilities(self):
        for c in self.profile:
            for v in self.profile[c]:
                self.profile[c][v] *= c.cost
        return self

    def cost_to_score_utilities(self):
        for c in self.profile:
            for v in self.profile[c]:
                self.profile[c][v] /= c.cost * 1.0
        return self

    def read_from_files(self, pattern : str): #assumes Pabulib data format
        cnt = 0
        for filename in Path(".").glob(pattern):
            cnt += 1
            cand_id_to_obj = {}
            with open(filename, 'r', newline='', encoding="utf-8") as csvfile:
                section = ""
                header = []
                reader = csv.reader(csvfile, delimiter=';')
                subunit = None
                meta = {}
                for i, row in enumerate(reader):
                    if len(row) == 0: #skip empty lines
                        continue
                    if str(row[0]).strip().lower() in ["meta", "projects", "votes"]:
                        section = str(row[0]).strip().lower()
                        header = next(reader)
                    elif section == "meta":
                        field, value  = row[0], row[1].strip()
                        meta[field] = value
                        if field == "subunit":
                            subunit = value
                            self.subunits.add(subunit)
                        if field == "budget":
                            self.budget += int(value.split(",")[0])
                    elif section == "projects":
                        project = {}
                        for it, key in enumerate(header[1:]):
                            project[key.strip()] = row[it+1].strip()
                        c_id = row[0]
                        c = Candidate(c_id, int(project["cost"]), project["name"], subunit=subunit)
                        self.profile[c] = {}
                        cand_id_to_obj[c_id] = c
                    elif section == "votes":
                        vote = {}
                        for it, key in enumerate(header[1:]):
                            vote[key.strip()] = row[it+1].strip()
                        v_id = row[0]
                        v_age = vote.get("age", None)
                        v_sex = vote.get("sex", None)
                        v = Voter(v_id, v_sex, v_age)
                        self.voters.add(v)
                        v_vote = [cand_id_to_obj[c_id] for c_id in vote["vote"].split(",")]
                        v_points = [1 for c in v_vote]
                        if meta["vote_type"] == "ordinal":
                            v_points = [int(meta["max_length"]) - i for i in range(len(v_vote))]
                        elif "points" in vote:
                            v_points = [int(points) for points in vote["points"].split(",")]
                        v_vote_points = zip(v_vote, v_points)
                        for (vote, points) in v_vote_points:
                            self.profile[vote][v] = points
        if cnt == 0:
            raise Exception("Invalid pattern: 0 files found")
        for c in set(c for c in self.profile):
            if c.cost > self.budget or sum(self.profile[c].values()) == 0: #nobody voted for the project; usually means the project was withdrawn
                del self.profile[c]

        return self
    


# Helper function for MES from PABULIB
def _mes_epsilons_internal(e : Election, eps_cost : bool, endow : dict[Voter, float], W : set[Candidate]) -> set[Candidate]:
    costW = sum(c.cost for c in W)
    remaining = e.profile.keys() - W
    rho = {c : c.cost - sum(endow[i] for i in e.profile[c]) for c in remaining}
    assert all(rho[c] > 0 for c in remaining)
    cnt = 0
    while True:
        cnt += 1
        next_candidate = None
        lowest_rho = math.inf
        voters_sorted = sorted(e.voters, key=lambda i: endow[i])
        for c in sorted(remaining, key=lambda c: rho[c]):
            if costW + c.cost > e.budget:
                continue
            if rho[c] >= lowest_rho:
                break
            sum_supporters = sum(endow[i] for i in e.profile[c])
            price = c.cost - sum_supporters
            for i in voters_sorted:
                if i not in e.profile[c]:
                    continue
                if endow[i] >= price:
                    if eps_cost:
                        rho[c] = price / c.cost
                    else:
                        rho[c] = price
                    break
                price -= endow[i]
            if rho[c] < lowest_rho:
                next_candidate = c
                lowest_rho = rho[c]
        if next_candidate is None:
            break
        else:
            W.add(next_candidate)
            costW += next_candidate.cost
            remaining.remove(next_candidate)
            for i in e.voters:
                if i in e.profile[next_candidate]:
                    endow[i] = 0
                else:
                    endow[i] -= min(endow[i], lowest_rho)
    return W

def _mes_internal(e : Election, real_budget : int = 0) -> (dict[Voter, float], set[Candidate]):
    W = set()
    costW = 0
    remaining = set(c for c in e.profile)
    endow = {i : 1.0 * e.budget / len(e.voters) for i in e.voters}
    rho = {c : c.cost / sum(e.profile[c].values()) for c in e.profile}
    while True:
        next_candidate = None
        lowest_rho = float("inf")
        remaining_sorted = sorted(remaining, key=lambda c: rho[c])
        for c in remaining_sorted:
            if rho[c] >= lowest_rho:
                break
            if sum(endow[i] for i in e.profile[c]) >= c.cost:
                supporters_sorted = sorted(e.profile[c], key=lambda i: endow[i] / e.profile[c][i])
                price = c.cost
                util = sum(e.profile[c].values())
                for i in supporters_sorted:
                    if endow[i] * util >= price * e.profile[c][i]:
                        break
                    price -= endow[i]
                    util -= e.profile[c][i]
                rho[c] = price / util
                if rho[c] < lowest_rho:
                    next_candidate = c
                    lowest_rho = rho[c]
        if next_candidate is None:
            break
        else:
            W.add(next_candidate)
            costW += next_candidate.cost
            remaining.remove(next_candidate)
            for i in e.profile[next_candidate]:
                endow[i] -= min(endow[i], lowest_rho * e.profile[next_candidate][i])
            if real_budget: #optimization for 'increase-budget' completions
                if costW > real_budget:
                    return None
    return endow, W

def _is_exhaustive(e : Election, W : set[Candidate]) -> bool:
    costW = sum(c.cost for c in W)
    minRemainingCost = min([c.cost for c in e.profile if c not in W], default=math.inf)
    return costW + minRemainingCost > e.budget

#MES Pabulib
def equal_shares(e : Election, completion : str = None) -> set[Candidate]:
    endow, W = _mes_internal(e)
    if completion is None:
        return W
    if completion == 'binsearch':
        initial_budget = e.budget
        while not _is_exhaustive(e, W): #we keep multiplying budget by 2 to find lower and upper bounds for budget
            b_low = e.budget
            e.budget *= 2
            res_nxt = _mes_internal(e, real_budget=initial_budget)
            if res_nxt is None:
                break
            _, W = res_nxt
        b_high = e.budget
        while not _is_exhaustive(e, W) and b_high - b_low >= 1: #now we perform the classical binary search
            e.budget = (b_high + b_low) / 2.0
            res_med = _mes_internal(e, real_budget=initial_budget)
            if res_med is None:
                b_high = e.budget
            else:
                b_low = e.budget
                _, W = res_med
        e.budget = initial_budget
        return W
    if completion == 'utilitarian_greedy':
        return _utilitarian_greedy_internal(e, W)
    if completion == 'phragmen':
        return _phragmen_internal(e, endow, W)
    if completion == 'add1':
        initial_budget = e.budget
        while not _is_exhaustive(e, W):
            e.budget *= 1.01
            res_nxt = _mes_internal(e, real_budget=initial_budget)
            if res_nxt is None:
                break
            _, W = res_nxt
        e.budget = initial_budget
        return W
    if completion == 'add1_utilitarian':
        initial_budget = e.budget
        while not _is_exhaustive(e, W):
            e.budget *= 1.01
            res_nxt = _mes_internal(e, real_budget=initial_budget)
            if res_nxt is None:
                break
            _, W = res_nxt
        e.budget = initial_budget
        return _utilitarian_greedy_internal(e, W)
    if completion == 'eps':
        return _mes_epsilons_internal(e, False, endow, W)
    assert False, f"""Invalid value of parameter completion. Expected one of the following:
        * 'binsearch',
        * 'utilitarian_greedy',
        * 'phragmen',
        * 'add1',
        * 'add1_utilitarian',
        * 'eps',
        * None."""
    
## The following is coded by us

def greedy(e : Election, u : bool = False) -> set[Candidate]:
    spent = 0
    selected_projects = []
    selected_row_info = []
    profiles = e.profile
    cands = profiles.keys()
    profs = [(c, c.cost, len(profiles[c])) for c in cands]
    if u: 
        profs = [((x[0]), x[1], x[1] * x[2]) for x in profs]
    sorted = profs
    sorted.sort(key = lambda votes: votes[2])
    while len(sorted) > 0:
        active = sorted.pop()
        if spent + active[1] < e.budget:
            selected_projects.append(active[0])
            selected_row_info.append(active)
            spent += active[1]

    return(set(selected_projects))

def onemin(e : Election, g : set = None, m = None, u : bool = False, ) -> set[Candidate]:
    unselected_proj = set(e.profile.keys())
    selected_proj = set()
    if g == None:
        g = {(x,) for x in e.voters}
    unsat_g = g.copy()
    spent = 0
    
    
    while len(unsat_g) > 0:
        null_can = Candidate(None, 0)
        best_proj = (null_can, 0)
        for proj in unselected_proj:
            if proj.cost <= e.budget - spent:
                groups_satisfying = 0
                satisfied_voters = set(e.profile[proj].keys())
                for gr in list(unsat_g):
                    if not satisfied_voters.isdisjoint(set(gr)): groups_satisfying += 1
                    
                if groups_satisfying / proj.cost > best_proj[1]:
                    best_proj = (proj, groups_satisfying / proj.cost)
        if best_proj[0] == null_can:
            break
        selected_proj.add(best_proj[0])
        unselected_proj.remove(best_proj[0])
        spent += best_proj[0].cost
        
        to_remove = set()
        for g in unsat_g:
            pres = False
            for v in e.profile[best_proj[0]].keys():
                if v in g:
                    pres = True
            if pres:
                to_remove.add(g)

        unsat_g -= to_remove

    if m == None:
        pass 
    else:
        unselected_profiles = {}
        for key in unselected_proj:
            unselected_profiles[key] = e.profile[key]
        subelection = Election(
            voters=e.voters,
            profile=unselected_profiles,
            budget = e.budget - spent
        )
        if m == "g":
            add_proj = greedy(subelection, u)
        elif m == "mes":
            add_proj = equal_shares(subelection)
        selected_proj = selected_proj.union(add_proj)

    return selected_proj

def optimal(e : Election, u: bool) -> set[Candidate]:

    #preprocessing the election data
    num_alt = len(e.profile)

    keys = [key for key in e.profile.keys()]
    cost_proj = np.asarray([key.cost for key in keys])
    votes_proj = np.array([len(e.profile[key]) for key in keys])

    #defining variables, objective and constraints
    x = cp.Variable(num_alt, integer = True)
    
    if u:
        #maximize cost utility
        o = cp.Maximize(np.multiply(votes_proj, cost_proj) @ x)
    else:
        #maximize vote utility
        o = cp.Maximize(votes_proj @ x)
    
    c = [0 <= x, x <= 1, cost_proj @ x <= e.budget]

    prob = cp.Problem(o, c)

    #running the problem
    result = prob.solve()
    vals = x.value

    #parsing the results'
    res = set()
    for i in range(len(vals)):
        if vals[i] == 1:
            res.add(keys[i])

    return res

def bounded_utility(e: Election, g : set, u : bool = False, eps : int = .5) -> set[Candidate]:
    
    #preprocessing the election data
    num_alt = len(e.profile)

    keys = [key for key in e.profile.keys()]
    cost_proj = np.asarray([key.cost for key in keys])
    votes_proj = np.array([len(e.profile[key]) for key in keys])

    #preproccess groups somehow
    #get per group total number of people-approvals of each project in a vector
    groups = list(g)

    gp = [0] * num_alt
    gv = [gp] * len(groups)
    for i in range(len(keys)):
        for v in e.profile[keys[i]].keys():
            for j in range(len(groups)):
                if v in groups[j]:
                    gv[j][i] += 1

    gz = []
    for group in groups:
        gz.append(len(group))
    
    

    #defining variables, objective and constraints
    x = cp.Variable(num_alt, integer = True)
    m = cp.Variable()
    
    if u:
        #maximize cost utility
        o = cp.Maximize(np.multiply(votes_proj, cost_proj) @ x)
    else:
        #maximize vote utility
        o = cp.Maximize(votes_proj @ x)
    
    c = [0 <= x, x <= 1, cost_proj @ x <= e.budget]

    #append group constraints to c

    for i in range(len(gv)):
        gp = gv[i]
        gs = gz[i]
        if u:
            c.append(np.multiply(gp, cost_proj) @ x / gs <= m)
            c.append(np.multiply(gp, cost_proj) @ x / gs >= eps * m)
        else:   
            c.append(gp @ x / gs <= m)
            c.append(gp @ x / gs >= eps * m)
        

    prob = cp.Problem(o, c)   

    #running the problem
    result = prob.solve()
    vals = x.value

    #parsing the results'
    res = set()
    for i in range(len(vals)):
        if vals[i] == 1:
            res.add(keys[i])

    return res

def eval_outcome (o : set[Candidate], e : Election, g : set):
    results = {}
    ##need to port code

    #money spent
    results["spent"] = sum([c.cost for c in o])

    #number of uncovered individuals
    #number of uncovered groups
    groups = list(g)
    nvotes = len(e.voters)
    ngroups = len(groups)
    
    covered_individuals = set()

    for proj in o:
       covered_individuals = covered_individuals.union(set(e.profile[proj].keys()))

    uncovered_voters = set(e.voters).difference(covered_individuals)
    results["uncovered_voters"] = uncovered_voters
    results["num_uncovered_voters"] = len(uncovered_voters)

    uncovered_groups = set()
    for group in groups:
        if not any(v in covered_individuals for v in g):
            uncovered_groups.add(group)
    results["uncovered_groups"] = uncovered_groups
    results["num_uncovered_groups"] = len(uncovered_groups)

    #utility by group by measure
    #todo

    #total utility by measure

    total_cost_u = 0
    total_vote_u = 0

    for proj in o:
        for v in e.profile[proj].keys():
            total_cost_u += proj.cost
            total_vote_u += 1

    results["cost_u"] = total_cost_u
    results["vote_u"] = total_vote_u

 
    return results

def generate_groups (e : Election, m : str):
    groups = set() 
    vs = list(e.voters)                              
    if m == "fl":
        maxi = len(vs) - 1
        groups.add(tuple(vs[0:(maxi // 2)]))
        groups.add(tuple(vs[(maxi // 2 + 1):maxi]))
     

    ##need to define methods and implement
    elif m == "agglom":
        clustering = AgglomerativeClustering(5)
        for c in e.profile:
            for v in vs:
                if v in e.profile[c]:
                    v.utilities.append(1)
                else:
                    v.utilities.append(0)
        voterListSorted = sorted(list(e.voters), key=lambda x: x.id)
        data = [voter.utilities for voter in voterListSorted]
        groupDict = defaultdict(set)
        labels = clustering.fit_predict(data)
        for index, label in enumerate(labels):
            groupDict[label].add(voterListSorted[index])
        for val in groupDict.values():
            groups.add(tuple(val))
        
    
    elif m == "age":
        ageList = [voter for voter in vs if voter.age]
        groups.add(tuple(voter for voter in ageList if int(voter.age) < 25))
        groups.add(tuple(voter for voter in ageList if int(voter.age) >= 25 and int(voter.age) <35))
        groups.add(tuple(voter for voter in ageList if int(voter.age) >= 35 and int(voter.age) <45))
        groups.add(tuple(voter for voter in ageList if int(voter.age) >= 45 and int(voter.age) <55))
        groups.add(tuple(voter for voter in ageList if int(voter.age) >= 55 and int(voter.age) <65))
        groups.add(tuple(voter for voter in ageList if int(voter.age) >= 65 and int(voter.age) <75))
        groups.add(tuple(voter for voter in ageList if int(voter.age) >= 75))
    elif m == "gend":
        groups.add(tuple(voter for voter in vs if voter.sex == "M"))
        groups.add(tuple(voter for voter in vs if voter.sex == "K"))
    else:
        raise Exception("Input Valid Partition Method")
    ##groups should be a set of tuples of voters
    return groups

def run_gauntlet(filename):
    e = Election().read_from_files(filename)
    

    
    groups = generate_groups(e, "fl")
    methods = {}
    #methods["opt_cost_u"] = optimal(e, True)
    #methods["opt_vote_u"] = optimal(e, False)
    #methods["bu_2_group_vote_u"] = bounded_utility(e, groups)
    #methods["greedy_vote_u"] = greedy(e)
    #methods["greedy_cost_u"] = greedy(e, True)
    #methods["mes"] = equal_shares(e)
    #methods["onemin_full"] = onemin(e)
    methods["onemin_group"] = onemin(e, groups, "mes")

    print(eval_outcome(methods["onemin_group"], e, groups))

    return methods

m = run_gauntlet("poland_warszawa_2022_ursynow.pb.txt")

print(m)
def run_test_groups(filename):
    e = Election().read_from_files(filename)
    # gend = generate_groups(e, "gend")
    # for group in gend:
    #     for ind in group:
    #         print(ind.sex)
    # age = generate_groups(e, "age")
    # for group in age:
    #     print(group[0].age)
    agglom = generate_groups(e, "agglom")
    for group in agglom:
        print(group)
    return 

# run_gauntlet("poland_warszawa_2022_ursynow.pb.txt")
run_test_groups("poland_warszawa_2022_ursynow.pb.txt")

