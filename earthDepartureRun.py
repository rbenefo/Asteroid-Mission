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
    sim.plotOn = False
    dEtaArr = np.arange(5, 95, 5)
    dVarr = []
    feasibilityArr = []
    timeArr = []
    for i in range(len(dEtaArr)):
        print("----------")
        print("dEta:", dEtaArr[i])
        print("----------")
        constants.dEta = dEtaArr[i]
        sim.init_problem(spacecraft, None, constants)
        feasible, deltaV, ret = sim.run()
        dVarr.append(ret["dV"])
        timeArr.append(ret["ToF"])
        feasibilityArr.append(feasible)
    timeArr = np.array(timeArr)[feasibilityArr]
    dVarr = np.array(dVarr)[feasibilityArr]/1000
    dEtaArr = np.array(dEtaArr)[feasibilityArr]
    fig,ax = plt.subplots(figsize = (14, 6))

    ax.plot(dEtaArr / 2, dVarr, color = "red", label = "Delta V")
    ax.set_xlabel("Critical Eccentric Anomaly (°)")
    ax.set_ylabel("Earth Departure Delta V (km/s)")
    ax2 = ax.twinx()


    ax2.set_ylabel("Earth Departure Time of Flight (days)")
    ax2.plot(dEtaArr / 2, timeArr, color = "blue", label="Time of Flight")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    plt.show()

def plotISPTradeoff(constants, spacecraft, sim):
    #Issue, this assumes a constant thrust, a useless assumption... need to model thrust relationship with ISP
    sim.plotOn = False
    ispArr = np.arange(200.0,2000.0, 20)
    tArr = []
    dProp = []
    for isp in ispArr:
        spacecraft.Isp = isp

        sim.init_problem(spacecraft, None, constants)
        feasible, deltaV, ret = sim.run()
        tArr.append(ret["ToF"])
        dProp.append(ret["dPropellant"])
        print(len(tArr))
        # print(tArr, dProp)
    fig,ax = plt.subplots(2)
    ax[0].set_title("ISP Tradeoff")
    ax[0].plot(ispArr, dProp, color = "orange", label = "Consumed propellant")
    ax[0].set_xlabel("ISP")
    ax[0].set_ylabel("Propellant consumed (kg)")
    ax2 = ax[0].twinx()
    ax2.plot(ispArr, tArr, color = "red", label="Flight time")
    ax2.set_ylabel("Time of flight (days)")
    ax2.legend()
    ax[0].legend()

    ax[1].plot(tArr, dProp)
    ax[1].set_xlabel("Time of flight (days)")
    ax[1].set_ylabel("Propellant consumed (kg)")
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

    constants = Constants("2023-05-20 23:59:54.003", "2023-06-10 23:59:54.003", dEta = 30)
    
    # engine = Engine("Hydros-M", isp = 310, tmax = 1.2)
    engine = Engine("MR-11G", isp = 219, tmax = 1.8)

    spacecraft = SpaceCraft("Endeavor", 87.0, engine)

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