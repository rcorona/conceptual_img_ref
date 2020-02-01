import sys

def extract_types(attr_path):
    """
    Return a dictionary with attribute indexes
    for each of the 5 attribute types in CUB.
    """
    
    lines = [line.strip() for line in open(attr_path, 'r')]
    attrs = [l.split()[-1] for l in lines]
    
    types = {
        'color': [],
        'shape': [],
        'pattern': [],
        'size': [],
        'length': []
    }
             
    for i in range(len(attrs)):
        t = attrs[i].split('::')[0].split('_')[-1]
        types[t].append(i)

    return types
    
if __name__ == '__main__':
    types = extract_types('attributes.txt')
