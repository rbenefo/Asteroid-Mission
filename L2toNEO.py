# Imports
import pykep as pk
import pygmo as pg
import numpy as np

# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pykep.examples import add_gradient
from common import Engine, Target, Constants, SpaceCraft


class Sim:
    def __init__(self, solver, years, plotOn = True):
        self.uda = solver
        self.uda.set_numeric_option("tol",1)
        self.algo = pg.algorithm(self.uda)
        self.years = years
        self.udp = None
        self.plotOn = plotOn
        self.sc = None

    def init_problem(self, spacecraft, target, constants):
        self.sc = spacecraft
        self.udp = add_gradient(pk.trajopt.lt_margo(
            target=target.target,
            n_seg=constants.n_seg,
            grid_type = "nonuniform", 
            t0 = [constants.starting_time, constants.ending_time], 
            tof = [200, 365.25*self.years], 
            m0 = spacecraft.m0,
            Tmax = spacecraft.Tmax,
            Isp = spacecraft.Isp,
            earth_gravity = False, 
            sep = True, 
            start = "earth"), 
            with_grad=True)

    def run(self):
        assert self.udp is not None, "You need to initialize the problem!"
        self.prob = pg.problem(self.udp)
        print(self.prob)
        self.prob.c_tol = [1e-5] * self.prob.get_nc()
        # self.pop = pg.population(self.prob, 1)
        # self.pop = self.algo.evolve(self.pop)
        # top = pg.fully_connected()
        pop = pg.population(self.prob, size = 1)
        isl = pg.island(algo = self.algo, pop = pop)
        archi = pg.archipelago(prob = self.prob)
        archi.push_back(isl)
        self.algo.set_verbosity(1)
        # print(get_island_count(archi))
        archi.evolve()
        # print(archi)
        archi.wait()
        feasibility_arr = self.eval_solution(archi)
        feasible = np.any(feasibility_arr)
        champions_x_copy = np.copy(archi.get_champions_x())
        champions_x_copy[np.where(feasibility_arr == 0)] = -1
        mf_arr = champions_x_copy[:,2]
        mf_arr_max = np.argmax(mf_arr)
        champion_x = champions_x_copy[mf_arr_max]

        if feasible and self.plotOn:
            self.plot(champion_x)
        output = self.udp.udp_inner.pretty(champion_x)

        mf = champion_x[2]
        mi = self.sc.m0
        isp = self.sc.Isp

        deltaV = isp * pk.G0 * np.log(mi / mf)
        return feasible, deltaV
        
    def eval_solution(self, archi):
        champions_x = np.array(archi.get_champions_x())
        feasibility_list = []
        for champion in champions_x:
            if self.prob.feasibility_x(champion):
                feasibility_list.append(1)
            else:
                feasibility_list.append(0)
        return np.array(feasibility_list)
    
    def plot(self, champions_x):
        axis = self.udp.udp_inner.plot_traj(champions_x, plot_segments=True)
        # 7 - plot control
        self.udp.udp_inner.plot_dists_thrust(champions_x)
        # plt.ion()
        plt.show()

def get_island_count(archi):
    count = 0
    for island in archi:
        count += 1
    return count

if __name__ == "__main__":
    sim = Sim(pg.ipopt(), 4, plotOn = True)
    ## 1032 mission
    mpcorbline = "10302   19.54  0.15 K224O 304.50803  183.39237  104.33136    4.37809  0.1363893  0.68687808   1.2721831  0 E2022-GJ2   556  11 1989-2022 0.71 M-v 3Ek MPCLINUX   0804  (10302) 1989 ML            20220414"
    mpcorbline = "K14Y00D 24.3   0.15 K1794 105.44160   34.12337  117.64264    1.73560  0.0865962  0.88781021   1.0721510  2 MPO369254   104   1  194 days 0.24 M-v 3Eh MPCALB     2803          2014 YD            20150618"
    mpcorbline = "l8784   25.60  0.15 K224Q  55.38838  289.95665  207.89998    2.10432  0.1398072  0.97449390   1.0075887  0 MPO685460   141  10 2012-2021 0.68 M-v 3Ek Pan        0803 (478784) 2012 UV136         20210603"

    target = Target("l8784", mpcorbline)
    # engine = Engine("Rit muX", isp = 3000, tmax = 0.0017)
    engine = Engine("Rit muX", isp = 2150, tmax = 0.0011)

    constants = Constants("2023-05-20 23:59:54.003", "2023-08-25 23:59:54.003")
    spacecraft = SpaceCraft("Endeavor", 50, engine)

    sim.init_problem(spacecraft, target, constants)
    feasible, deltaV = sim.run()
    print(feasible, deltaV)
