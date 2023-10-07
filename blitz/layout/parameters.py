FILE_PARAMS = [
    {'name': 'Load as 8 bit',
     'type': 'bool',
     'value': False},
    {'name': 'Size ratio',
     'type': 'float',
     'value': 1,
     'limits': (0, 1),
     'step': 0.1},
    {'name': 'Subset ratio',
     'type': 'float',
     'value': 1,
     'limits': (0, 1),
     'step': 0.1},
    {'name': 'max. RAM (GB)',
     'type': 'float',
     'value': 1.0,
     'step': 0.1,
     'limits': (0.1 , 1)}
]

ROI_PARAMS = [
    {'name': 'Pixels', 'type': 'float', 'value': 1},
    {'name': 'in mm', 'type': 'float', 'value': 1},
    {'name': 'show in mm', 'type': 'bool', 'value': False},
]
