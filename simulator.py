#!/usr/bin/env python
# coding: utf-8

get_ipython().system(' pip install -q -r requirements.txt')


import math, random, logging
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s %(filename)s] %(message)s')

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
    def __init__(self, population, area, 
                 resident_mobility_freq_cohorts:List[datetime.timedelta],
                 resident_mobility_dist_cohorts:List[Tuple[float, float]]):
        """
        :param resident_mobility_freq_cohorts: common frequencies at which residents move out during shelter in place like scenarios
        :param resident_mobility_dist_cohorts: common distances residents travel during shelter in place like scenarios
        """
        self.population, self.area = population, area
        self.length = self.breadth = math.sqrt(area)
        self.rm_frequencies, self.rm_travel_distances = resident_mobility_freq_cohorts, resident_mobility_dist_cohorts
        
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
    def __init__(self, name:str, rh:ResidentHealth, 
                 move_freq:datetime.timedelta, move_dist_range:Tuple[float, float],
                 ec:EpidemicConfig, cc:CityConfig
                ):
        """
        :param move_freq: how frequently does the resident go out of their house
        :param move_dist_range: when they go out, what distance range do they travel
        """
        self.name = name
        self.infected, self.infected_since = HealthStateEnum.NEVER_INFECTED, SENTINAL_DATE
        self.recovered_on, self.died_on = SENTINAL_DATE, SENTINAL_DATE
        self.health, self.ec, self.cc = rh, ec, cc
        self.move_freq, self.move_dist_range= move_freq, move_dist_range
        self.loc_hist:List[Tuple[datetime.datetime, float, float]] = []
    
    def _valid_coords(self, x:float, y:float):
        """Check if resident is at a valid location"""
        min_x, min_y, max_x, max_y = 0, 0, self.cc.length, self.cc.breadth
        return min_x <= x <= max_x and min_y <= y <= max_y

    def location(self):
        return self._x_pos, self._y_pos
    
    def move_to(self, new_x:float, new_y:float, date:datetime.datetime):
        if self._valid_coords(new_x, new_y):
            self._x_pos, self._y_pos = new_x, new_y
            self.loc_hist.append((date, new_x, new_y))
            return True
        return False
    
    def __repr__(self):
        return f"{self.name} is at ({self._x_pos}, {self._y_pos})"
    
    def __call__(self):
        infected_since = self.infected_since.strftime("%c") if self.infected_since < SENTINAL_DATE else None
        return self._x_pos, self._y_pos, self.name, self.infected, infected_since
    
    def is_infected(self):
        return self.infected == HealthStateEnum.INFECTED
    
    def is_alive(self):
        return self.infected != HealthStateEnum.DEAD
    
    def infect(self, date:datetime.datetime):
        self.infected = HealthStateEnum.INFECTED
        self.infected_since = date
        
    def _did_recover(self):
        return self.health.recovery_prob > random.random() # TODO: Any better way to build this?
        
    def cure(self, date:datetime.datetime, epidemic_config:EpidemicConfig):
        """Returns true if the resident was successfully cured"""
        duration_since_infected = date - self.infected_since
        if self.is_infected() and duration_since_infected >= epidemic_config.recover_after:
            if self._did_recover():
                self.infected = HealthStateEnum.RECOVERED
                self.recovered_on = date
                return True
            if duration_since_infected > epidemic_config.decease_after:
                self.kill(date)
        return False
        
    def kill(self, date:datetime.datetime):
        self.infected = HealthStateEnum.DEAD
        self.died_on = date
        
    def progress_time(self, current_date:datetime.datetime, move_prob:float=1) -> bool:
        """
        Make necessary changes to the state of the resident
        :param move_prob: for finer control over move_freq. 
            Say resident goes / moves out once a week and we progress the simulator day by day, 
            move_prob can be used to select the day of move during the week
        :return True if state of the resident changed or False otherwise
        """
        last_moved_on, _, _ = self.loc_hist[-1]
        if last_moved_on + self.move_freq < current_date and random.random() <= move_prob and self._move_to_rand(current_date):
            return True # as the resident moved this time
        return False
        
    def _move_to_rand(self, date:datetime.datetime, max_attempts:int=3) -> bool:
        """
        :param max_attempts: max number of times to try to find a valid new position before giving up
        :returns True if the resident was able to move to a valid position
        """
        for attempt_num in range(max_attempts):
            distance, direction = random.uniform(*self.move_dist_range), random.uniform(0, 2 * math.pi)
            new_x, new_y = self._x_pos + distance * math.cos(direction), self._y_pos + distance * math.sin(direction)
            move_ok = self.move_to(new_x, new_y, date)
            if move_ok:
                return True
        return False


class EpidemicCurve:
    def __init__(self):
        self.trend:List[Dict[str, float]] = []
            
    def log_stats(self, date:datetime, stats:Dict[str, float]):
        log = {"date": date}
        log.update(stats)
        self.trend.append(log)
    
    def visualize_time_series(self, stats:List[str], rename_cols:Dict[str, str]=dict(), options:opts=opts()):
        """
        Plots the given stats over a period of a time
        :param stats: names of statistics to plot
        :param rename_cols: any human readable names for the statistics
        :param options: plotting options for holoviews
        """
        df = pd.DataFrame(self.trend).fillna(0)
        stats = list(map(lambda s: rename_cols.get(s, s), df.columns & stats))
        df = df.rename(columns=rename_cols)
        point_curves = [hv.Scatter(df[["date", statistic]], label=statistic) for statistic in stats]
        line_curves = [hv.Curve(points) for points in point_curves]
        return (hv.Overlay(line_curves + point_curves)).                    opts(opts.Scatter(tools=["hover"], size=6)).                    opts(padding=0.05, height=375, legend_position="bottom", title="").                    opts(options)


class City:
    def __init__(self, cc:CityConfig, ec:EpidemicConfig):
        self.cc, self.epidemic_config = cc, ec
        self.population, self.approx_infected_count = cc.population, ec.approx_infected_count
        self.current_date = ec.start_date
        self.residents = self.load_population_info()
        self.allocate_homes()
        
    def load_population_info(self):
        infected_prob = self.approx_infected_count / self.population
        residents = [] # TODO: initialize array of self.population size
        for i in range(self.population):
            rh = ResidentHealth(np.random.uniform(0.5, 1)) # TODO: Need a better way to compile resident's health
            resident_attrs = {
                "name": f"{i}",
                "rh": rh,
                "move_freq": random.choice(self.cc.rm_frequencies), 
                "move_dist_range": random.choice(self.cc.rm_travel_distances),
                "ec": self.epidemic_config,
                "cc": self.cc
            }
            resident = Resident(**resident_attrs)
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

    def allocate_homes(self, max_attempts_per_resident:int=3):
        """
        :param max_attempts_per_resident: max number of times to try and assign a home to a resident before erroring out
        """
        # for resident in progress_bar(self.residents):
        for resident in self.residents:
            allocated = False
            for _ in range(max_attempts_per_resident):
                x, y = random.random() * self.cc.length, random.random() * self.cc.breadth
                allocated = resident.move_to(x, y, self.current_date)
                if allocated:
                    break
                logging.warning(f"({x}, {y}) as home for resident {resident.name} was not accepted. Trying again...")
            if not allocated:
                raise RuntimeError(f"Unable to allocate home for resident, {resident}")
    
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
    
    def progress_time(self, by:datetime.timedelta) -> Dict[str, int]:
        """Returns some importance changes that happened during this period"""
        # TODO: cure, spread and progress_time each loop over the list of residents at least once which is expensive
        recovered_cases, deceased_cases = self.cure_disease()
        new_cases = self.spread_disease()
        residents_state_changed = list(filter(lambda r: r.progress_time(self.current_date), self.residents))
        self.current_date += by
        return {
            "num_recovered_cases": len(recovered_cases), 
            "num_deceased_cases": len(deceased_cases), 
            "num_new_cases": len(new_cases), 
            "num_state_changed": len(residents_state_changed),
            "state_changed_sample": np.random.choice(residents_state_changed, min(10, len(residents_state_changed)))
        }

    def visualize(self):
        cmap = {HealthStateEnum.NEVER_INFECTED: "green", HealthStateEnum.RECOVERED: "blue", HealthStateEnum.INFECTED: "red", HealthStateEnum.DEAD: "gray"}
        
        plot_df = self.residents_as_df()
        plot_df["color"] = plot_df["infected"].apply(lambda health: cmap[health])
        label = f"On {self.current_date.strftime('%c')}"
        
        population_scatter = hv.Points(
            plot_df, kdims=["_x_pos", "_y_pos"], label=label,
            vdims=["name", 'infected', "infected_since", "move_freq", "move_dist_range", "color"]).opts(
            size=4, tools=["hover"], padding=0.05, color="color", xaxis=None, yaxis=None, 
            legend_position="bottom")
        return population_scatter


class Simulator:
    def __init__(self, city:City, start_date:datetime):
        self.city = city
        self.curve = EpidemicCurve()
        self.current_date = start_date
        self.curve.log_stats(start_date, self.city_stats())
    
    def city_stats(self, other_stats:Dict[str, Union[int, float]]=dict()):
        health_stats = dict(sunnyvale.city_health().to_dict(orient="split")["data"])
        return {**other_stats, **health_stats}
        
    def progress_time(self, inc_by:datetime.timedelta):
        changes = self.city.progress_time(inc_by)
        self.current_date += inc_by
        self.curve.log_stats(self.current_date, self.city_stats(changes))
    
    def visualize(self):
        daily_plot_cols_labels = {"INFECTED": "Infected", "RECOVERED": "Recovered"}
        change_plot_cols_labels = {
            "num_recovered_cases": "# newly recovered",
            "num_deceased_cases": "# of new deaths",
            "num_new_cases": "# of new cases"
        }
        mobility_plot_cols_labels = {
            "num_state_changed": "# of residents moved" # NOTE: Only people moving is considered a state change as of today
        }
        gspec = pn.GridSpec(sizing_mode='stretch_both')
        mobility_plot = self.curve.visualize_time_series(mobility_plot_cols_labels.keys(), 
            rename_cols=mobility_plot_cols_labels, 
            options=opts(title="∆ in number of residents moved", ylabel="Periodic change", height=300, width=500, show_legend=False))
        daily_plot = self.curve.visualize_time_series(daily_plot_cols_labels.keys(), 
            rename_cols=daily_plot_cols_labels, 
            options=opts(ylabel="Total residents as of date", width=475))
        change_plot = self.curve.visualize_time_series(change_plot_cols_labels.keys(), 
            rename_cols=change_plot_cols_labels,
            options=opts(ylabel="Periodic change in metric", width=475))
        
        gspec[0, :2] = (self.city.visualize() + mobility_plot).opts(shared_axes=False)
        gspec[1, :2] = (daily_plot + change_plot).opts(shared_axes=False)
        return gspec


start_date = datetime.datetime.strptime("2020-03-01", "%Y-%m-%d")
resident_mobility_freq_cohorts = [datetime.timedelta(days=1), datetime.timedelta(days=2), datetime.timedelta(days=7)]
resident_mobility_dist_cohorts = [(1, 2), (2, 3), (2, 5)]

sunnyvale_config = CityConfig(100, 144, resident_mobility_freq_cohorts, resident_mobility_dist_cohorts)
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




