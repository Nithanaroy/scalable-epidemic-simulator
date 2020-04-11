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

from typing import Dict, Any, Callable, List, Union

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


SENTINAL_DATE = datetime.datetime(2300, 1, 1)


class Resident:
    def __init__(self, name:str, x_pos=-1, y_pos=-1):
        self.x_pos, self.y_pos, self.name = x_pos, y_pos, name
        self.infected, self.infected_since = HealthStateEnum.NEVER_INFECTED, SENTINAL_DATE
    
    def location(self):
        return self.x_pos, self.y_pos
    
    def __repr__(self):
        return f"{self.name} is at ({self.x_pos}, {self.y_pos})"
    
    def __call__(self):
        infected_since = self.infected_since.strftime("%c") if self.infected_since < SENTINAL_DATE else None
        return self.x_pos, self.y_pos, self.name, self.infected, infected_since
    
    def is_infected(self):
        return self.infected == HealthStateEnum.INFECTED
    
    def infect(self, date:datetime.datetime):
        self.infected = HealthStateEnum.INFECTED
        self.infected_since = date
        
    def cure(self):
        self.infected = HealthStateEnum.RECOVERED


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
                 recover_after=datetime.timedelta(days=2)):
        self.approx_infected_count, self.start_date = approx_infected_count, start_date
        self.infection_radius = infection_radius
        self.recover_after = recover_after
        
    def __repr__(self):
        str_repr = {
            "start_date": lambda date_obj: datetime.datetime.strftime(date_obj, "%c")
        }
        return dict_as_str(self.__dict__, str_repr)


class EpidemicCurve:
    def __init__(self):
        self.trend:List[Dict[str, float]] = []
            
    def log_stats(self, date:datetime, total_infected:int):
        self.trend.append({
            "date": date,
            "infected": total_infected
        })
    
    def visualize(self):
        df = pd.DataFrame(self.trend)
        return (
            hv.Curve(df[["date", "infected"]]) *\
            hv.Points(df[["date", "infected"]])
        ).opts(opts.Points(tools=["hover"], size=6)).opts(padding=0.05)


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
        residents = [Resident(name=f"{i}") for i in range(self.population)]
        for person in residents:
            if random.random() <= infected_prob:
                person.infect(self.epidemic_config.start_date)
        return residents
    
    def residents_as_df(self):
        return pd.DataFrame([p.__dict__ for p in self.residents])

    def allocate_homes(self):
        # for person in progress_bar(self.residents):
        for person in self.residents:
            person.x_pos, person.y_pos = random.random() * self.length, random.random() * self.breadth
    
    def spread_disease(self) -> List[Resident]:
        new_cases:List[Resident] = []
        # TODO: can be slow for large cities
        # for p1 in progress_bar(self.residents):
        for p1 in self.residents:
            for p2 in self.residents:
                if p1 != p2 and p1.is_infected():
                    # ASSUMPTION: An infected person cannot be infected again
                    if not p2.is_infected():
                        social_distance = distance.euclidean(p1.location(), p2.location())
                        if social_distance < self.epidemic_config.infection_radius:
                            new_cases.append(p2)
                        
        for person in new_cases:
            person.infect(self.current_date)
        return new_cases
    
    def cure_disease(self) -> List[Resident]:
        recovered_cases:List[Resident] = []
        for person in self.residents:
            if person.is_infected() and self.current_date - person.infected_since >= self.epidemic_config.recover_after:
                person.cure()
                recovered_cases.append(person)
        return recovered_cases
    
    def next_day(self, by:datetime.timedelta) -> int:
        recovered_cases = self.cure_disease()
        new_cases = sunnyvale.spread_disease()
        self.current_date += by
        return len(recovered_cases), len(new_cases)

    def visualize(self):
        cmap = {HealthStateEnum.NEVER_INFECTED: "green", HealthStateEnum.RECOVERED: "blue", HealthStateEnum.INFECTED: "red"}
        plot_df = self.residents_as_df()
        plot_df["color"] = plot_df["infected"].apply(lambda health: cmap[health])
        label = f"On {self.current_date.strftime('%c')}"
        return hv.Points(
            plot_df, kdims=["x_pos", "y_pos"], vdims=["name", 'infected', "infected_since", "color"], label=label).opts(
            size=4, tools=["hover"], padding=0.05, color="color", xaxis=None, yaxis=None, 
            legend_position="bottom"
        )


class Simulator:
    def __init__(self, city:City, start_date:datetime):
        self.city = city
        self.curve = EpidemicCurve()
        self.curve.log_stats(start_date, len(list(filter(lambda person: person.is_infected(), self.city.residents))))
        self.current_date = start_date
        
    def progress_time(self, inc_by:datetime.timedelta):
        num_recovered_cases, num_new_cases = self.city.next_day(inc_by)
        self.current_date += inc_by
        self.curve.log_stats(self.current_date, len(list(filter(lambda person: person.is_infected(), self.city.residents))))
    
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




