test = {   'name': 'MaxPool Layer',
    'points': 15,
    'suites': [   {   'cases': [   {'code': '>>> list(MaxPool2d(3)(torch.zeros((10,3,64,64))).shape) == [10,3,21,21]\nTrue', 'hidden': False, 'locked': False, 'points': 0},
                                   {'code': '>>> type(MaxPool2d(3)(torch.zeros((10,3,64,64)))) in [torch.Tensor]\nTrue', 'hidden': False, 'locked': False, 'points': 0}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
