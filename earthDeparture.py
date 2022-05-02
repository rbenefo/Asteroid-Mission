# Imports
from calendar import c
from pickletools import read_float8
import pykep as pk
import pygmo as pg
import numpy as np

from pykep.core import epoch
# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pykep.examples import add_gradient

from numpy.linalg import norm
import sys

class EarthDeparture:
    def __init__(self,
                target = None,
                t0=[pk.epoch(8000), pk.epoch(9000)],
                m0 = 20,
                tf = [10, 80],
                Tmax=0.0017,
                Isp=3000.0,
                dEta = 30):
        """Decision vector (chromosome)::
        [t0, tf], tf is length of the final coast to the target in days
        """
        self.target = target
        self.dEta = np.deg2rad(dEta)
        self.__earth = pk.planet.jpl_lp('earth')
        self.__sc = pk.sims_flanagan.spacecraft(m0, Tmax, Isp)

        self.__interplanetary_mu = pk.MU_SUN

        self.dt = 1000 #in seconds

        lb = [t0[0].mjd2000]
        ub = [t0[1].mjd2000]

        self.__lb = lb
        self.__ub = ub


    def get_bounds(self):
        return self.__lb, self.__ub


    def _propagate(self, x, plot = False):
        dVs = []
        thrustTimes = []

        #Burns to reach escape velocity
        ##Frame of reference is EARTH
        earth_radius = 6779000 #in meters
        a = 250*1000 + earth_radius
        b = 35786*1000 + earth_radius
        ri = [0, a, 0] #~35786 km GTO
        xp = [ri[0]]; yp = [0]; zp =[0]
        tx = []; ty = []; tz = []
        
        speedi = self.orbitAtAltitude(a, b)
        # speedi = self.orbitAtAltitude(ri)

        vi = [speedi, 0,0]
        mi = self.__sc.mass
        time = 0
        stopThrusting = False
        while True:
            currEta = pk.ic2par(ri, vi, self.__earth.mu_self)[5]
            #Burn
            if self.isInBurnZone(currEta) and not stopThrusting:
                thrust = self.__sc.thrust * np.array(vi) / norm(np.array(vi))
                veff = self.__sc.isp * pk.G0
                rf, vf, mf = pk.propagate_taylor(ri, vi, mi, thrust, self.dt, self.__earth.mu_self, veff, -15, -15)
                # sys.exit()
                deltaV = self.__sc.isp * pk.G0 * np.log(mi / mf)
                dVs.append(deltaV) #function of mf and mi and ISP
                tx.append(rf[0]); ty.append(rf[1]); tz.append(rf[2])
                thrustTimes.append(time)
            #Coast
            else:
                rf, vf = pk.propagate_lagrangian(ri, vi, self.dt, self.__earth.mu_self)
                if stopThrusting:
                    if np.abs(norm(vf) - norm(vi))  < 0.3:
                        break

            xp.append(rf[0]); yp.append(rf[1]); zp.append(rf[2])
            mi = mf
            vi = vf
            ri = rf
            time += self.dt

            if self.reachedEscapeVelocity(vf, rf):
                stopThrusting = True

        #Convert spacecraft frame of reference to sun-centered
        t1 = epoch(x[0]+time * (1/pk.DAY2SEC))
        r_earth, v_earth = self.__earth.eph(t1)
        rf = np.array(rf)
        rf += np.array(r_earth)
        vf = np.array(vf)
        vf += np.array(v_earth)
        rf = tuple(rf)
        vf = tuple(vf)

        # #Final powered burn to target, followed by coast, and final correction burn
        # dt = x[1] * pk.DAY2SEC
        # cw = False #to correct...
        # r_earth_future, v_earth_future = self.__earth.eph(epoch(x[0]+(dt+time) * (1/pk.DAY2SEC)))
        # r_target = [r * 1.01 for r in r_earth_future]
        # v_target = [v * 1.01 for v in v_earth_future]
        # l = pk.lambert_problem(rf, r_target, dt, self.__interplanetary_mu, cw)
        # v_end_l = l.get_v2()[0]
        # v_beg_l = l.get_v1()[0]

        # DV1 = norm([a - b for a, b in zip(v_beg_l, vf)])
        # DV2 = norm([a - b for a, b in zip(v_end_l, v_target)])

        if plot:
            fig = plt.figure(figsize = (8,6))
            ax1 = fig.gca(projection='3d')

            ax1.set_title("Earth Departure")
            ax1.scatter(xp,yp,zp, s = 1, label= "Coasting")
            ax1.scatter(tx,ty,tz, c = "r", label="Thrusting")
            ax1.set_xlim([-0.1e9, 0.1e9])
            ax1.set_ylim([-1.2e9, 0])
            ax1.set_box_aspect([1,1,1])
            # ax1.scatter([0], [0], [0], c = "b", s = 100) #Earth

            ax1.legend()
            
            # fig = plt.figure(figsize = (8,6))
            # ax2 = fig.gca(projection = "3d")
            # ax2.set_title("Interplanetary Transfer")
            # pk.orbit_plots.plot_lambert(l, color='r', legend=True, units=1.0, axes=ax2)
            # t2 = epoch(x[0]+(dt+time) * (1/pk.DAY2SEC))
            # pk.orbit_plots.plot_planet(self.__earth, t0=t1, color=(0.8, 0.8, 1), legend=True, units=1.0, axes=ax2)
            # ax2.scatter(r_target[0], r_target[1], r_target[2], c = "b", s = 20, label="L2 {}".format(repr(t2)[:10])) #L2
            # ax2.scatter([0], [0], [0], c = "y", s = 30) #Sun

            # ax2.legend()

            fig = plt.figure(figsize = (6,8))
            ax3 = fig.gca()
            ax3.set_title("Earth Departure Thrusting Times")
            ax3.set_xlabel("Mission Day")
            thrustTimes = np.array(thrustTimes)/60 / 60 / 24
            ones = np.ones(thrustTimes.shape)
            ax3.bar(thrustTimes, ones)
            ax3.set_ylim([0,1])

        # return time, DV1, DV2, dVs
        return time, dVs

    def fitness(self, x):
        # time, DV1, DV2, dVs= self._propagate(x)
        time, dVs = self._propagate(x)
        # totDv = np.sum(np.array(dVs)) + DV1 + DV2
        totDv = np.sum(np.array(dVs))
        # dt = x[1] * pk.DAY2SEC
        # totTime = dt+time
        # score  = [totDv + totTime/60/60/24*2]
        score = [totDv]
        # print("Escape Time", time/60/60/24/365)
        return score

    def orbitAtAltitude(self, a, b):
        speed = np.sqrt(self.__earth.mu_self*2*b / (a*(a+b)))
        return speed
    
    def isInBurnZone(self, eta):
        if eta <= self.dEta / 2 and eta >= -self.dEta / 2:
            return True
        else:
            return False

    def reachedEscapeVelocity(self, v, r):
        escapeVel = np.sqrt(2*self.__earth.mu_self/norm(r))
        return norm(v) >= escapeVel

    def pretty(self, x):
        t0 = x[0]
        # tf = x[1]
        # time, DV1, DV2, dVs = self._propagate(x)
        time, dVs = self._propagate(x)
        # T = (time / pk.DAY2SEC) + tf
        T = time / pk.DAY2SEC
        t_arr = T + t0

        # deltaV = DV1 +DV2 + np.sum(np.array(dVs))
        deltaV = np.sum(np.array(dVs))
        mf = self.__sc.mass / np.exp(np.sum(np.array(dVs))/(self.__sc.isp *pk.G0))

        dPropellant = self.__sc.mass - mf
        print("Departure:", pk.epoch(t0), "(", t0, "mjd2000 )")
        print("Time of flight:", round(time / pk.DAY2SEC , 2), "days")
        print("Arrival:", pk.epoch(t_arr), "(", t_arr, "mjd2000 )")
        # print("L2 burn Delta-v:", round(DV1 +DV2, 2), "m/s")
        print("Climb Delta-v:", round(np.sum(np.array(dVs)), 2), "m/s")
        # print("Total Delta-v:", round(deltaV, 2), "m/s")
        print("Propellant consumption:", round(dPropellant, 2), "kg")


        ret = {}
        ret["departure"] = t0
        ret["ToF"] = T
        ret["arrival"] = t_arr
        ret["dV"] = deltaV
        ret["mf"] = mf
        ret["dPropellant"] = dPropellant
        return ret



    def plot_traj(self, x):
        self._propagate(x, plot = True)
    

    def get_name(self):
        return "Earth Departure Trajectory"

    def __repr__(self):
        return self.get_name()

