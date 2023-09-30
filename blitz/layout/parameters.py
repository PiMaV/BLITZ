FILE_PARAMS = [
    {'name': 'Loading Pars',
        'type': 'group', 'children': [
        {'name': 'Load as 8 bit', 'type': 'bool', 'value': False},
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
    ]},
    {'name': 'Load TOF Data',
        'type': 'group',
        'expanded' :False,
        'tooltip': 'TOF Data to load',
        'children': [
        {'name': 'Browse', 'type': 'action'},
        {'name': 'Show', 'type': 'bool', 'value': False, 'enabled': False},
    ]},
]

EDIT_PARAMS = [
    {'name': 'Calculations',
        'type': 'group',
        'tooltip': 'Various Calculations on the full set',
        'children': [
        {'name': 'Operation',
            'type': 'list',
            'values': ['Org', 'Min', 'Max', 'Mean', 'STD'],
            'value': 'Org'}
    ]},
    {'name': 'Manipulations',
        'type': 'group',
        'tooltip': 'Manipulation Tools',
        'children': [
        {'name': 'Rotate CCW', 'type': 'bool', 'value': False},
        {'name': 'Flip X', 'type': 'bool', 'value': False},
        {'name': 'Flip Y', 'type': 'bool', 'value': False},
    ]},
    {'name': 'Mask',
        'type': 'group',
        'children': [
        {'name': 'Mask', 'type': 'bool', 'value': False},
        {'name': 'Apply Mask', 'type': 'action'},
    ]},
    {'name': 'ROI',
        'type': 'group',
        'tooltip': 'Region of Interest',
        'children': [
        {'name': 'Enable', 'type': 'bool', 'value': False},
        {'name': 'Pixels', 'type': 'float', 'value': 1},
        {'name': 'in mm', 'type': 'float', 'value': 1},
        {'name': 'show in mm', 'type': 'bool', 'value': False},
    ]},
    {'name': 'Visualisations',
        'type': 'group',
        'tooltip': 'Alternate apperance',
        'children': [
        {'name': 'Crosshair', 'type': 'bool', 'value': True},
    ]},
]
