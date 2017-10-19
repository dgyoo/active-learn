from collections import Iterable

def arg_changes(argparser, arg, ignore=[]):
    arg_dict = vars(arg)
    changes_dict = {}
    for key in arg_dict:
        val0 = argparser.get_default(key)
        val = arg_dict[key]
        if val0 != val and not key in ignore:
            changes_dict[key] = val
    changes = ''
    for k in sorted(changes_dict):
        v = changes_dict[k]
        if isinstance(v, bool):
            v = int(v)
        elif isinstance(v, int):
            v = str(v)
        elif isinstance(v, float):
            v = '{:.2f}'.format(v)
        elif isinstance(v, str):
            pass
        else:
            raise Exception('Not supported argument value: {}'.format(v))
        changes += '{}={},'.format(k, v)
    return changes[:-1]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0 and isinstance(val, Iterable):
            self.sum = [0 for _ in val]
            self.avg = [0 for _ in val]
        self.val = val
        self.count += n
        if isinstance(val, Iterable):
            for i, v in enumerate(val):
                self.sum[i] += v * n
                self.avg[i] = self.sum[i] / self.count
        else:
            self.sum += val * n
            self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0
        assert path.endswith('.log'), 'File extension should be ''log''.'

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistant number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.')
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log

    def max(self, column_index=2):
        log = self.read()
        return max([values[column_index] for values in log])

def to_string(values, precision='%.2f'):
    if not isinstance(values, Iterable):
        return precision % values
    string = ''
    for v in values:
        string += (precision % v) + ','
    return string[:-1]
