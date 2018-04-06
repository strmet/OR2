import subprocess
import os
import time
import datetime
import paramiko
import argparse
import getpass

# Writing down all the options:

dataset_numbers = [str(i+1) for i in range(29)]
interfaces = ['cplex']  # , 'docplex']
cs = [str(7)]  # [str(i) for i in range (7,10)]
rins_options = [7]  # [-1, 0, 7, 100]
rins_options = [str(el) for el in rins_options]

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

server = "login.dei.unipd.it"
username = str(input("Please, enter your username (@" + server + "): "))
pwd = str(getpass.getpass(prompt=username+"@" + server + ": "))

current_filename = ''
p_time = 500
for d in dataset_numbers:
    for i in interfaces:
        for c in cs:
            for r in rins_options:
                # Formatting/constructing the instruction to be given:
                instruction = "python2.7 "+ project_path + "/src/__init__.py"

                # Options to be added:
                instruction += " --dataset " + d
                instruction += " --interface " + i
                instruction += " --C " + c
                instruction += " --rins " + r
                instruction += " --polishtime " + str(p_time)
                # The following three instructions are needed to construct the folder's name
                current_time = time.time()
                timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M:%S')
                current_folder = "run__" + timestamp + "__d=" + d + "i=" + i + "c=" + c + "rins=" + r
                instruction += " --outfolder " + current_folder
                # Setting the timeout and saving the output to a log file:
                instruction += " --timeout 600 > "
                instruction += project_path + '/out' + '/' + current_folder + '/cplex_log.log'
                instruction = instruction.strip()

                # Create such folder, if it doesn't exist
                if not os.path.exists(project_path + '/out' + '/' + current_folder):
                    os.makedirs(project_path + '/out' + '/' + current_folder)

                # Creating a new ssh instance
                ssh = paramiko.SSHClient()
                # Required, since "login.dei.unipd.it" is not a "well known ssh host"
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                # Connect to login.dei.unipd.it
                ssh.connect(server, username=username, password=pwd)
                # Give this job to the cluster
                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(instruction)
