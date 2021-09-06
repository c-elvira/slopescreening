# -*- coding: utf-8 -*-
import json

import numpy as np


class Setup(object):
   """docstring for Setup"""
   def __init__(self, setup_id):
      super(Setup, self).__init__()
      self.setup_id = setup_id
      
      with open('../setups.json') as json_file:
         data = json.load(json_file)[f"setup{setup_id}"]

      # 
      self.m = data["m"]
      self.n = data["n"]

      self.list_dic = data["dictionaries"]

      self.normalize = data["normalize"]

      self.n_rep = data["n_rep"]

      self.list_sequence = data["sequences"]
      self.list_ratio_lbd = data["list_ratio_lbd"]

      self.nb_dic         = len( self.list_dic )
      self.nb_sequence    = len( self.list_sequence )
      self.nb_ratio_lbd   = len( self.list_ratio_lbd )