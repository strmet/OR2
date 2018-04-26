#!/usr/bin/env python3
import os
import time
import datetime
import paramiko
import getpass

'''
    FIRST TIME INFO
    
    Log in to your account with ssh and clone the repository inside the home folder
    
'''

# All the options for the job
dataset_numbers = [1, 7, 16, 20, 26]
interfaces = ['cplex']
rins_options = [-1, 0, 10, 100]
num_iterations = 5

server = "login.dei.unipd.it"

# Creating a new ssh instance
ssh = paramiko.SSHClient()
# Required, since "login.dei.unipd.it" is not a "well known ssh host"
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Ask the user the password
pwd = str(getpass.getpass(prompt="Password: "))

if getpass.getuser() == 'met': # Avoid asking username
    username = "stringherm"
elif getpass.getuser() == 'venir':
    username = "venirluca"
else:
    username = str(input("Please, enter your username (@" + server + "): "))
    pwd = str(getpass.getpass(prompt=username+"@" + server + ": "))

# Connect to server
ssh.connect(server, username=username, password=pwd)

# Open secure file transfer protocol instance
sftp = ssh.open_sftp()  # per trasferire i file

# Remote project path
remote_path = "/home/" + username + "/OR2/"

# Get parent directory name
local_path = os.path.dirname(os.getcwd()) + "/"

# Create custom folder
current_time = time.time()
timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M:%S')
current_folder = "run__" + timestamp

try:
    sftp.mkdir(remote_path + "out/")
except IOError:
    pass

sftp.mkdir(remote_path + "out/" + current_folder)

# Files to be uploaded
files = ['src/__init__.py', 'src/lib/WindFarm.py', 'src/lib/callback.py']

# Create a local file that will be sent to the server
with open("commands.job", "w") as fp:
    fp.write("#!/bin/bash \n")
    fp.write("export PYTHONPATH=$PYTHONPATH:/nfsd/opt/CPLEX12.6/cplex/python/3.4/x86-64_linux/ \n")

    for it in range(num_iterations):
        for d in dataset_numbers:
            for i in interfaces:
                for r in rins_options:
                    # Formatting/constructing the instruction to be given:
                    instruction = "python3 " + remote_path + "src/__init__.py"

                    # Options to be added:
                    instruction += " --dataset " + str(d)
                    instruction += " --interface " + i
                    instruction += " --rins " + str(r)
                    instruction += " --cluster "

                    instruction += " --outfolder " + current_folder
                    # Setting the timeout and saving the output to a log file:
                    instruction += " --timeout 300 \n"

                    fp.write(instruction)

print("Copying files")
for file in files:
    file_remote = remote_path + file
    file_local = local_path + file

    print(file_local + ' >>> ' + file_remote)
    try:
        sftp.remove(file_remote)
    except IOError:
        print("File was not on the cluster")
        pass

    sftp.put(file_local, file_remote)

# Put the file on the current folder on the cluster and delete the local one
print(local_path + 'scripts/commands.job' ' >>> ' + remote_path + 'out/' + current_folder + '/commands.job')
sftp.put(local_path + 'scripts/commands.job', remote_path + 'out/' + current_folder + '/commands.job')
os.remove("commands.job")

# Create the results file
file = sftp.file(remote_path + 'out/' + current_folder + '/results.csv', "w", -1)
file.write("{0},".format(len(interfaces)*len(rins_options)))
for i in interfaces:
    for r in rins_options:
        if d == dataset_numbers[-1] and i == interfaces[-1] and r == rins_options[-1]:
            file.write("{0}_rins{1}".format(i, r))
        else:
            file.write("{0}_rins{1},".format(i, r))
file.write("\n")

file.flush()

# Give this job to the cluster
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("export SGE_ROOT=/usr/share/gridengine \n" +
                                                     "cd {0}out/{1} \n".format(remote_path, current_folder) +
                                                     "qsub -cwd commands.job")
# dev'essere "tutto assieme"
#  o si dimentica dell'export

# qsub -cwd == "current working directory". DEVE ESSERE MESSO PRIMA!!!

# Print output and errors
print(ssh_stdout.read().decode('utf-8'))
print(ssh_stderr.read().decode('utf-8'))

sftp.close()
ssh.close()
