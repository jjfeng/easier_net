#!/usr/bin/env scons

import os

from os.path import join
from nestly.scons import SConsWrap
from nestly import Nest
import SCons.Script as sc

# Command line options

sc.AddOption('--output', type='string', help="output folder", default='_output')
sc.AddOption('--clusters', type='string', help="cluster to use", default='beagle')

env = sc.Environment(
        ENV=os.environ,
        clusters=sc.GetOption('clusters'),
        output=sc.GetOption('output'))

sc.Export('env')

env.SConsignFile()

flag = 'simulation_deconstruct'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'simulation_support_prob'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'cumida'
sc.SConscript(flag + '/sconscript', exports=['flag'])
