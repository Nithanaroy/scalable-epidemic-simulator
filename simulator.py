#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install -q -r requirements.txt')


# In[1]:


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

from typing import Dict, Any, Callable

hv.extension('bokeh')
renderer = hv.renderer('bokeh')
output_notebook()


# In[2]:


def dict_as_str(d:Dict[str, Any], custom_str_repr:Dict[str, Callable]={}):
    """
    Represent a dictionary as a formatted string
    Useful for representing class objects for quick inspection
    """
    return json.dumps({attr: custom_str_repr.get(attr, lambda i: i)(val) for attr, val in d.items()}, indent=2)    


# In[3]:


class Resident:
    def __init__(self, name:str, x_pos=-1, y_pos=-1, infected=False, immunity_level:int=1):
        self.x_pos, self.y_pos, self.name = x_pos, y_pos, name
        self.infected = infected
    
    def location(self):
        return self.x_pos, self.y_pos
    
    def __repr__(self):
        return f"{self.name} is at ({self.x_pos}, {self.y_pos})"
    
    def __call__(self):
        return self.x_pos, self.y_pos, self.name, self.infected
    
    def infect(self):
        self.infected = True
        
    def cure(self):
        self.infected = False


# In[4]:


class CityConfig:
    def __init__(self, population=0, area=0):
        self.population, self.area = population, area
        
    def __repr__(self):
        return dict_as_str(self.__dict__)


# In[5]:


class EpidemicConfig:
    def __init__(self, approx_infected_count=0, date:datetime=datetime.datetime(2020, 3,1)):
        self.approx_infected_count, self.date = approx_infected_count, date
        
    def __repr__(self):
        str_repr = {
            "date": lambda date_obj: datetime.datetime.strftime(date_obj, "%c")
        }
        return dict_as_str(self.__dict__, str_repr)


# In[6]:


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


# In[7]:


class City:
    def __init__(self, config:CityConfig, ec:EpidemicConfig):
        self.population, self.approx_infected_count = config.population, ec.approx_infected_count
        self.length = self.breadth = math.sqrt(config.area)
        infected_prob = self.approx_infected_count / self.population
        self.people = [Resident(name=f"{i}", infected=random.random() <= infected_prob) for i in range(self.population)]
        self.allocate_homes()

    def allocate_homes(self):
        # for person in progress_bar(self.people):
        for person in self.people:
            person.x_pos, person.y_pos = random.random() * self.length, random.random() * self.breadth
    
    def spread_disease(self, infection_radius:float):
        new_cases:List[Resident] = []
        # TODO: can be slow for large cities
        # for p1 in progress_bar(self.people):
        for p1 in self.people:
            for p2 in self.people:
                if p1 != p2 and p1.infected:
                    social_distance = distance.euclidean(p1.location(), p2.location())
                    if social_distance < infection_radius:
                        new_cases.append(p2)
                        
        for person in new_cases:
            person.infect()
    
    def next_day(self, by:datetime.timedelta):
        sunnyvale.spread_disease(0.5)

    def visualize(self):
        people = hv.Points([p() for p in self.people], vdims=["name", 'infected']).opts(
            size=4, tools=["hover"], padding=0.05, color='infected', cmap="Dark2", 
            xaxis=None, yaxis=None, legend_position="bottom"
        )
        return people


# In[8]:


class Simulator:
    def __init__(self, city:City, start_date:datetime):
        self.city = city
        self.curve = EpidemicCurve()
        self.curve.log_stats(start_date, len(list(filter(lambda person: person.infected, self.city.people))))
        self.current_date = start_date
        
    def progress_time(self, inc_by:datetime.timedelta):
        self.city.next_day(inc_by)
        self.current_date += inc_by
        self.curve.log_stats(self.current_date, len(list(filter(lambda person: person.infected, self.city.people))))
    
    def visualize(self):
        return (self.city.visualize() + self.curve.visualize()).opts(shared_axes=False)


# In[9]:


start_date = datetime.datetime.strptime("2020-03-01", "%Y-%m-%d")
sunnyvale_config = CityConfig(100, 144)
corona_config = EpidemicConfig(5, start_date)
sunnyvale = City(sunnyvale_config, corona_config)


# In[10]:


sunnyvale_toy_simulator = Simulator(sunnyvale, start_date)


# In[11]:


sunnyvale_toy_simulator.visualize()


# In[12]:


sunnyvale_toy_simulator.progress_time(datetime.timedelta(days=1))
sunnyvale_toy_simulator.visualize()


# In[14]:


sunnyvale_toy_simulator.progress_time(datetime.timedelta(days=1))
sunnyvale_toy_simulator.visualize()


# ## Rough

# In[ ]:


e = EpidemicCurve()
e.log_stats(datetime.datetime.strptime("2020-03-01", "%Y-%m-%d"), 20)
e.log_stats(datetime.datetime.strptime("2020-03-02", "%Y-%m-%d"), 30)
e.log_stats(datetime.datetime.strptime("2020-03-03", "%Y-%m-%d"), 35)

e.visualize()


# In[ ]:


hv.Points(range(10)) + hv.Points(range(20, 10, -1)).opts(shared_axes=False)


# In[ ]:




