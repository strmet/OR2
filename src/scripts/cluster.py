import subprocess
import os


# Writing down all the options:

dataset_numbers = [str(i+1) for i in range(30)]
interfaces = ['cplex', 'docplex']
cs = [str(i) for i in range (7,10)]
rins_options = [str(i) for i in range(5,10)]

# Searching for the absolute path of the home of this project
abspath = os.path.abspath(os.path.dirname(__file__)).strip()
path_dirs = abspath.split('/')
path_dirs.remove('')

project_path = ''
or2_found = False
i = 0
while not or2_found:
    if path_dirs[i] == 'OR2':
        or2_found = True
    project_path += '/' + path_dirs[i]
    i += 1

out_path = project_path + '/out'
current_filename = ''
for d in dataset_numbers:
    for i in interfaces:
        for c in cs:
            for r in rins_options:
                instruction = "python3.6 ../__init__.py"
                instruction += " --dataset " + d
                instruction += " --interface " + i
                instruction += " --C " + c
                instruction += " --rins " + r
                instruction += " --timeout 300 > "
                instruction += out_path + '/'
                current_filename = "run_on_d" + d + "i_" + i + "c" + c + "rins" + r + ".log"
                instruction += current_filename
                instruction = instruction.strip()
                subprocess.call(instruction, shell=True)
