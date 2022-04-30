# Imports
import pykep as pk
import pygmo as pg
import numpy as np

# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pykep.examples import add_gradient
from earthDeparture import EarthDeparture
from common import Engine, Target, Constants, SpaceCraft
import sys




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
        self.udp = add_gradient(EarthDeparture(
            t0 = [constants.starting_time, constants.ending_time],
            m0 = spacecraft.m0,
            Tmax = spacecraft.Tmax,
            Isp = spacecraft.Isp,
            dEta = constants.dEta           
        ), with_grad = True)

    def run(self):
        assert self.udp is not None, "You need to initialize the problem!"
        self.prob = pg.problem(self.udp)
        self.prob.c_tol = [1e-5] * self.prob.get_nc()

        self.algo.set_verbosity(1)


        self.pop = pg.population(self.prob, 1)

        # We solve the problem
        self.pop = self.algo.evolve(self.pop)

        # print(get_island_count(archi))
        champion_x = self.pop.champion_x
        champion_f = self.pop.champion_f

        feasible = self.eval_solution(champion_x)
        print("Feasible?:", feasible)

        if feasible and self.plotOn:
            self.plot(champion_x)
        ret = self.udp.udp_inner.pretty(champion_x)

        mi = self.sc.m0
        isp = self.sc.Isp

        deltaV = isp * pk.G0 * np.log(mi / ret["mf"])
        return feasible, deltaV, ret
        
    def eval_solution(self, champion_x):
        # champions_x = np.array(archi.get_champions_x())
        # feasibility_list = []
        # for champion in champions_x:
        #     if self.prob.feasibility_x(champion):
        #         feasibility_list.append(1)
        #     else:
        #         feasibility_list.append(0)
        # return np.array(feasibility_list)
        return self.prob.feasibility_x(champion_x)
    
    def plot(self, champions_x):
        self.udp.udp_inner.plot_traj(champions_x)
        plt.show()


def runEtaTest(constants, spacecraft, sim):
    dEtaArr = np.arange(5, 95, 5)
    dVarr = []
    feasibilityArr = []
    for i in range(len(dEtaArr)):
        print("----------")
        print("dEta:", dEtaArr[i])
        print("----------")
        constants.dEta = dEtaArr[i]
        sim.init_problem(spacecraft, None, constants)
        feasible, deltaV = sim.run()
        dVarr.append(deltaV)
        feasibilityArr.append(feasible)

    dVarr = np.array(dVarr)[feasibilityArr]/1000
    dEtaArr = np.array(dEtaArr)[feasibilityArr]

    plt.plot(dEtaArr, dVarr)
    plt.show()

def plotISPTradeoff(constants, spacecraft, sim):
    ispArr = np.arange(200.0,2000.0,100)
    tArr = []
    dProp = []
    for isp in ispArr:
        spacecraft.Isp = isp

        sim.init_problem(spacecraft, None, constants)
        feasible, deltaV, ret = sim.run()
        tArr.append(ret["ToF"])
        dProp.append(ret["dPropellant"])
        print(tArr, dProp)
    fig,ax = plt.subplots()


    ax.set_title("ISP Tradeoff")
    ax.plot(ispArr, tArr, color = "red", label="Flight time")
    ax.set_xlabel("ISP")
    ax.set_ylabel("Time of flight (days)")
    ax2 = ax.twinx()
    ax2.plot(ispArr, dProp, color = "orange", label = "Consumed propellant")
    ax2.set_ylabel("Propellant consumed (kg)")

    fig2,ax3 = plt.subplots()
    ax.set_title("ISP Tradeoff")
    ax3.plot(tArr, dProp)
    ax3.set_xlabel("Time of flight (days)")
    ax3.set_ylabel("Propellant consumed (kg)")
    plt.show()

def test(constants, spacecraft, sim):
    sim.init_problem(spacecraft, None, constants)
    feasible, deltaV, ret = sim.run()


def calcMassPercentage(spacecraft, constants, sim):
    #Need at least 50kg dry mass available for the final stage
    sim.init_problem(spacecraft, None, constants)
    feasible, deltaV, ret = sim.run()
    remainingMass = spacecraft.m0 - ret["dPropellant"] - 50
    print("Remaining mass:", remainingMass)

if __name__ == "__main__":
    sim = Sim(pg.ipopt(), 3, plotOn = True)

    constants = Constants("2022-09-20 23:59:54.003", "2032-09-20 23:59:54.003", dEta = 15)
    
    engine = Engine("MR-11G", isp = 310, tmax = 1.2)

    spacecraft = SpaceCraft("Endeavor", 120.0, engine)

    # plotISPTradeoff(constants, spacecraft, sim)
    test(constants, spacecraft, sim)
    # runEtaTest(constants, spacecraft, sim)

    # dEtaArr = np.arange(5, 95, 5)
    # dVarr = []
    # feasibilityArr = []
    # for i in range(len(dEtaArr)):
    #     print("----------")
    #     print("dEta:", dEtaArr[i])
    #     print("----------")
    #     constants.dEta = dEtaArr[i]
    #     sim.init_problem(spacecraft, None, constants)
    #     feasible, deltaV = sim.run()
    #     dVarr.append(deltaV)
    #     feasibilityArr.append(feasible)

    # dVarr = np.array(dVarr)[feasibilityArr]/1000
    # dEtaArr = np.array(dEtaArr)[feasibilityArr]

    # plt.plot(dEtaArr, dVarr)
    # plt.show()



    #To do:
        #test changes in Eta
        #test changes in time weighting in score??
        #prettify plots?