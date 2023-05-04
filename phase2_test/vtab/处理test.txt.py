from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent

base_path = Path('/data/dataset/Place_365_Standard').resolve()

k, q = 0, 'test'
with open(this_directory / f'Places_LT_{q} copy.txt') as f:
        lines = f.readlines() 

with open(this_directory / f'Places_LT_{q}.txt', 'w') as f:
    new_lines = ['/'.join([l if i!=0 else f'{l}_large' for i, l in enumerate(line.split('/')) if i !=1]) for line in lines]
    error_list = [(i, line) for i, line in enumerate(new_lines) if not (base_path/line.split()[0]).exists()]
    print(len(error_list))
    f.writelines(new_lines)
        
qs = ['val', 'train']
ts = ['data_large', 'data_large']
def process_line(line):
    splits = line.split('/')
    return '/'.join([ts[k], splits[1][:1],  *splits[1:]])
for k, q in enumerate(qs):
    with open(this_directory / f'Places_LT_{q} copy.txt') as f:
        lines = f.readlines() 

    with open(this_directory / f'Places_LT_{q}.txt', 'w') as f:
        
        new_lines = [process_line(line) for line in lines]
        error_list = [(i, line) for i, line in enumerate(new_lines) if not (base_path/line.split()[0]).exists()]
        print(len(error_list))
        print(error_list[:2])
        f.writelines(new_lines)
    
