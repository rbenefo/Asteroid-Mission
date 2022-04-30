import pykep as pk

class Engine:
    def __init__(self, name, tmax, isp):
        self.isp = isp
        self.tmax = tmax
        self.name= name
    def __repr__(self):
        return "{}".format(self.name)



class Target:
    def __init__(self, name, mpcorb):
        self.name = name
        self.mpcorbline = mpcorb
        self.target = pk.planet.mpcorb(self.mpcorbline)

    def __repr__(self):
        return "{}".format(self.name)

class Constants:
    def __init__(self, start, end, dEta = 30):
        self.starting_time = pk.epoch_from_string(start)
        self.ending_time = pk.epoch_from_string(end)
        self.dEta = dEta
        self.n_seg = 30

    def setdEta(self, dEta):
        self.dEta = dEta

class SpaceCraft:
    def __init__(self, name, m0, engine):
        self.name = name
        self.m0 = m0
        self.Tmax = engine.tmax
        self.Isp = engine.isp

    def __repr__(self):
        return "{}".format(self.name)
