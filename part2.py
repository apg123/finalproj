import codecs
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv


# Classes from Pabulib
class Voter:
    def __init__(self,
            id : str,
            sex : str = None,
            age : int = None,
            subunits : set[str] = set()
            ):
        self.id = id #unique id
        self.sex = sex
        self.age = age
        self.subunits = subunits

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

def onemin(e : Election, g : list = None, m = None, u : bool = False, ) -> set[Candidate]:
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
            for v in e.profile[best_proj[0]]:
                if v in g:
                    pres = True
            if pres:
                to_remove.add(g)

        unsat_g -= to_remove
    
    ##todo: capstone after onemin

    return selected_proj
            


def optimal(e : Election, u: bool) -> set[Candidate]:
    pass
    ##todo

def bounded_utility(e: Election, g : list, u : bool, eps : int) -> set[Candidate]:
    pass
    ##todo after optimal

def eval_outcome (o : set[Candidate], e : Election, g):
    results = {}
    ##need to port code

    #money spent 

    #number of uncovered individuals

    #number of uncovered groups

    #utility by group by measure

    #total utility by measure

 
    return results

def generate_groups (e : Election, m : int):
    pass
    ##need to define methods and implement

    ##groups should be a set of tuples of voters

def run_gauntlet(filename):
    e = Election().read_from_files("poland_warszawa_2022_ursynow.pb.txt")

e = Election().read_from_files("poland_warszawa_2022_ursynow.pb.txt")
print(greedy(e))
print(onemin(e))