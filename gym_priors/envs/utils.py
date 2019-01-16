#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:15:19 2018

@author: molano
"""


import os
import difflib


def look_for_folder(main_folder='priors/', exp=''):
    data_path = ''
    possibilities = []
    for root, dirs, files in os.walk(main_folder):
        ind = root.rfind('/')
        possibilities.append(root[ind+1:])
        if root[ind+1:] == exp:
            data_path = root
            break

    if data_path == '':
        candidates = difflib.get_close_matches(exp, possibilities,
                                               n=1, cutoff=0.)
        print(exp + ' NOT FOUND IN ' + main_folder)
        if len(candidates) > 0:
            print('possible candidates:')
            print(candidates)

    return data_path


def list_str(l):
    nice_string = str(l[0])
    for ind_el in range(1, len(l)):
        nice_string += '_'+str(l[ind_el])
    return nice_string


def num2str(num):
    return str(int(num/1000))+'K'
