# -*- coding: utf-8 -*-
"""
Get variable name from variable
"""

import traceback
import re


def varname(var):
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    mat = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    print mat
    return mat
