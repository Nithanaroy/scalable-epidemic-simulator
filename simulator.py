#!/usr/bin/env python
# coding: utf-8

get_ipython().system(' pip install -q -r requirements.txt')


import math, random
from fastprogress import progress_bar
import holoviews as hv

import bokeh
from bokeh.plotting import show
from bokeh.io import output_notebook

# TODO: failing on Mac with Assertion failed: (PassInf && "Expected all immutable passes to be initialized")
# from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from holoviews.operation import decimate
from holoviews import opts, Cycle

import pandas as pd
from pathlib import Path
import param
import panel as pn
from functools import partial
import numpy as np
from subprocess import PIPE, run
from multiprocessing import cpu_count
import pandas as pd
from fastprogress import progress_bar
import numpy as np
from scipy.spatial import distance
import datetime
from pprint import pprint
import json

from typing import Dict, Any, Callable, List, Union, Tuple

hv.extension('bokeh')
renderer = hv.renderer('bokeh')
output_notebook()


def dict_as_str(d:Dict[str, Any], custom_str_repr:Dict[str, Callable]={}):
    """
    Represent a dictionary as a formatted string
    Useful for representing class objects for quick inspection
    """
    return json.dumps({attr: custom_str_repr.get(attr, lambda i: i)(val) for attr, val in d.items()}, indent=2)    


class HealthStateEnum:
    NEVER_INFECTED = 0
    INFECTED = 1
    RECOVERED = 2
    DEAD = 3
    
    def __init__(self):
        self.id_state_map = dict(map(lambda item: (item[1], item[0]) , HealthStateEnum.__dict__.items()))
    
    def state_for_id(self, id:int) -> str:
        return self.id_state_map[id]


SENTINAL_DATE = datetime.datetime(2300, 1, 1)


class CityConfig:
    def __init__(self, population=0, area=0):
        self.population, self.area = population, area
        
    def __repr__(self):
        return dict_as_str(self.__dict__)


class EpidemicConfig:
    def __init__(self, 
                 approx_infected_count=0, 
                 start_date:datetime=datetime.datetime(2020, 3,1),
                 infection_radius:float=0.5,
                 recover_after=datetime.timedelta(days=2),
                 decease_after=datetime.timedelta(days=7)):
        self.approx_infected_count, self.start_date = approx_infected_count, start_date
        self.infection_radius = infection_radius
        self.recover_after, self.decease_after = recover_after, decease_after
        
    def __repr__(self):
        str_repr = {
            "start_date": lambda date_obj: datetime.datetime.strftime(date_obj, "%c")
        }
        return dict_as_str(self.__dict__, str_repr)


class ResidentHealth:
    """
    Has information about resident's general health condition
    """
    def __init__(self, recovery_prob:float=0.5):
        # TODO: Compute recovery probability from known things like age, medical history etc.
        if not 0 < recovery_prob <= 1:
            raise AttributeError(f"Invalid value {recovery_prob}, for probability of recovery")
        self.recovery_prob = recovery_prob
    
    def __repr__(self):
        return dict_as_str(self.__dict__)


class Resident:
    def __init__(self, name:str, rh:ResidentHealth, x_pos=-1, y_pos=-1):
        self.x_pos, self.y_pos, self.name = x_pos, y_pos, name
        self.infected, self.infected_since = HealthStateEnum.NEVER_INFECTED, SENTINAL_DATE
        self.recovered_on, self.died_on = SENTINAL_DATE, SENTINAL_DATE
        self.health = rh
    
    def location(self):
        return self.x_pos, self.y_pos
    
    def __repr__(self):
        return f"{self.name} is at ({self.x_pos}, {self.y_pos})"
    
    def __call__(self):
        infected_since = self.infected_since.strftime("%c") if self.infected_since < SENTINAL_DATE else None
        return self.x_pos, self.y_pos, self.name, self.infected, infected_since
    
    def is_infected(self):
        return self.infected == HealthStateEnum.INFECTED
    
    def is_alive(self):
        return self.infected != HealthStateEnum.DEAD
    
    def infect(self, date:datetime.datetime):
        self.infected = HealthStateEnum.INFECTED
        self.infected_since = date
        
    def did_recover(self):
        return self.health.recovery_prob > random.random() # TODO: Any better way to build this?
        
    def cure(self, date:datetime.datetime, epidemic_config:EpidemicConfig):
        """Returns true if the resident was successfully cured"""
        duration_since_infected = date - self.infected_since
        if self.is_infected() and duration_since_infected >= epidemic_config.recover_after:
            if self.did_recover():
                self.infected = HealthStateEnum.RECOVERED
                self.recovered_on = date
                return True
            if duration_since_infected > epidemic_config.decease_after:
                self.kill(date)
        return False
        
    def kill(self, date:datetime.datetime):
        self.infected = HealthStateEnum.DEAD
        self.died_on = date


class EpidemicCurve:
    def __init__(self):
        self.trend:List[Dict[str, float]] = []
            
    def log_stats(self, date:datetime, stats:Dict[str, float]):
        log = {"date": date}
        log.update(stats)
        self.trend.append(log)
    
    def visualize(self):
        df = pd.DataFrame(self.trend).fillna(0)
        stats = df.columns.drop("date").drop("NEVER_INFECTED")
        line_curves = [hv.Curve(df[["date", statistic]], label=statistic) for statistic in stats]
        point_curves = [hv.Points(df[["date", statistic]]) for statistic in stats]
        return (hv.Overlay(line_curves + point_curves)).                    opts(opts.Points(tools=["hover"], size=6)).                    opts(padding=0.05, ylabel="City's Health", legend_position="bottom", height=375)


class City:
    def __init__(self, config:CityConfig, ec:EpidemicConfig):
        self.config, self.epidemic_config = config, ec
        self.population, self.approx_infected_count = config.population, ec.approx_infected_count
        self.length = self.breadth = math.sqrt(config.area)
        self.residents = self.load_population_info()
        self.allocate_homes()
        self.current_date = ec.start_date
        
    def load_population_info(self):
        infected_prob = self.approx_infected_count / self.population
        residents = [] # TODO: initialize array of self.population size
        for i in range(self.population):
            resident_health = ResidentHealth(np.random.uniform(0.5, 1)) # TODO: Need a better way to compile resident's health
            resident = Resident(f"{i}", resident_health)
            if random.random() <= infected_prob:
                resident.infect(self.epidemic_config.start_date)
            residents.append(resident)
        return residents
    
    def residents_as_df(self):
        return pd.DataFrame([p.__dict__ for p in self.residents])
    
    def city_health(self):
        """Current state of health of residents in the city"""
        health_labels = HealthStateEnum()
        df = self.residents_as_df().infected.value_counts().reset_index().rename({"index": "Health State", "infected": "Count"}, axis=1)
        df["Health State"] = df["Health State"].apply(lambda health_type_id: health_labels.state_for_id(health_type_id))
        return df

    def allocate_homes(self):
        # for resident in progress_bar(self.residents):
        for resident in self.residents:
            resident.x_pos, resident.y_pos = random.random() * self.length, random.random() * self.breadth
    
    def spread_disease(self) -> List[Resident]:
        new_cases:List[Resident] = []
        # TODO: can be slow for large cities https://github.com/Nithanaroy/scalable-epidemic-simulator/issues/1
        # for p1 in progress_bar(self.residents):
        for p1 in self.residents:
            for p2 in self.residents:
                if p1 != p2 and p1.is_infected():
                    # ASSUMPTION: An infected resident cannot be infected again
                    if not p2.is_infected():
                        social_distance = distance.euclidean(p1.location(), p2.location())
                        if social_distance < self.epidemic_config.infection_radius:
                            new_cases.append(p2)
                        
        for resident in new_cases:
            resident.infect(self.current_date)
        return new_cases
    
    def cure_disease(self) -> List[Resident]:
        recovered_cases, deceased_cases = [], [] # Each is a List[Resident]
        for resident in self.residents:
            if resident.cure(self.current_date, self.epidemic_config):
                recovered_cases.append(resident)
            elif not resident.is_alive():
                deceased_cases.append(resident)
        return recovered_cases, deceased_cases
    
    def next_day(self, by:datetime.timedelta) -> int:
        recovered_cases = self.cure_disease()
        new_cases = self.spread_disease()
        self.current_date += by
        return len(recovered_cases), len(new_cases)

    def visualize(self):
        cmap = {HealthStateEnum.NEVER_INFECTED: "green", HealthStateEnum.RECOVERED: "blue", HealthStateEnum.INFECTED: "red", HealthStateEnum.DEAD: "gray"}
        
        plot_df = self.residents_as_df()
        plot_df["color"] = plot_df["infected"].apply(lambda health: cmap[health])
        label = f"On {self.current_date.strftime('%c')}"
        
        population_scatter = hv.Points(
            plot_df, kdims=["x_pos", "y_pos"], vdims=["name", 'infected', "infected_since", "color"], label=label).opts(
            size=4, tools=["hover"], padding=0.05, color="color", xaxis=None, yaxis=None, 
            legend_position="bottom")
        return population_scatter


class Simulator:
    def __init__(self, city:City, start_date:datetime):
        self.city = city
        self.curve = EpidemicCurve()
        self.current_date = start_date
        self.curve.log_stats(start_date, self.city_stats())
    
    def city_stats(self):
        return dict(sunnyvale.city_health().to_dict(orient="split")["data"])
        
    def progress_time(self, inc_by:datetime.timedelta):
        num_recovered_cases, num_new_cases = self.city.next_day(inc_by)
        self.current_date += inc_by
        self.curve.log_stats(self.current_date, self.city_stats())
    
    def visualize(self):
        return (self.city.visualize() + self.curve.visualize()).opts(shared_axes=False)


start_date = datetime.datetime.strptime("2020-03-01", "%Y-%m-%d")
sunnyvale_config = CityConfig(100, 144)
corona_config = EpidemicConfig(5, start_date)
sunnyvale = City(sunnyvale_config, corona_config)


sunnyvale_toy_simulator = Simulator(sunnyvale, start_date)

sunnyvale_toy_simulator.visualize()


sunnyvale_toy_simulator.progress_time(datetime.timedelta(days=1))
sunnyvale_toy_simulator.visualize()


e = EpidemicCurve()
e.log_stats(datetime.datetime.strptime("2020-03-01", "%Y-%m-%d"), 20)
e.log_stats(datetime.datetime.strptime("2020-03-02", "%Y-%m-%d"), 30)
e.log_stats(datetime.datetime.strptime("2020-03-03", "%Y-%m-%d"), 35)

e.visualize()


hv.Points(range(10)) + hv.Points(range(20, 10, -1)).opts(shared_axes=False)




