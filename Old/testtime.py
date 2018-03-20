# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:47:05 2018

@author: rkennedy
"""

def get_sec(time_str):
    h, m = time_str.split(':')
    result = int(h) * 3600 + int(m) * 60 
    return result

print(get_sec('23:45'))