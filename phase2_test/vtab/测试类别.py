from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent

base_path = Path('/data/dataset/Place_365_Standard').resolve()
qs = ['test', 'val', 'train']
for q in qs:
    with open(this_directory / f'Places_LT_{q} copy.txt') as f:
        lines = f.readlines() 
    classes = set([line.split('/')[1] for line in lines])
    print(len(classes))
with open(this_directory/'类别.txt', 'w') as f:
    f.writelines([f"{i}\n" for i in sorted(classes)])
    

classes = []
for i in (base_path/'data_large').glob('*'):
    if i.is_dir():
        print(i)
        for j in i.glob('*'):
            if j.is_dir():
                classes.append(j.name)
print(len(classes))
with open(this_directory/'类别2.txt', 'w') as f:
    f.writelines([f"{i}\n" for i in sorted(classes)])