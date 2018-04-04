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
                current_folder = "run_d=" + d + "i=" + i + "c=" + c + "rins=" + r
                instruction += " --outfolder " + current_folder
                instruction += " --timeout 60 > "
                instruction += project_path + '/out' + '/' + current_folder + '/cplex_log.log'
                instruction = instruction.strip()
                if not os.path.exists(project_path + '/out' + '/' + current_folder):
                    os.makedirs(project_path + '/out' + '/' + current_folder)
                subprocess.call(instruction, shell=True)
