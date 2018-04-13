import os
import time
import datetime
import paramiko


dataset_numbers = [30]
interfaces = ['cplex']  # , 'docplex']
cs = [8]  # [str(i) for i in range (7,10)]
rins_options = [7, 10, 20]  # [-1, 0, 7, 100]

server = "login.dei.unipd.it"

username = str(input("Please, enter your username (@" + server + "): "))
pwd = str(getpass.getpass(prompt=username+"@" + server + ": "))

# Creating a new ssh instance
ssh = paramiko.SSHClient()
# Required, since "login.dei.unipd.it" is not a "well known ssh host"
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# Connect to login.dei.unipd.it
ssh.connect(server, username=username, password=pwd)
# Give this job to the cluster
#ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("ls")

sftp = ssh.open_sftp()

files = ['src/__init__.py', 'src/lib/WindFarm.py', 'scripts/commands.job']
remote_path = "/home/" + username + "/OR2/"

local_path = os.path.dirname(os.path.realpath(__file__))
local_path = local_path[: -7]

current_time = time.time()
timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M:%S')
current_folder = "run__" + timestamp

sftp.mkdir(remote_path + "out/" + current_folder)

with open("commands.job", "a") as fp:
    fp.write("#!/bin/bash \n")
    fp.write("export PYTHONPATH=$PYTHONPATH:/nfsd/opt/CPLEX12.6/cplex/python/3.4/x86-64_linux/ \n")
    for d in dataset_numbers:
        for i in interfaces:
            for c in cs:
                for r in rins_options:
                    # Formatting/constructing the instruction to be given:
                    instruction = "python3 "+ remote_path + "src/__init__.py"

                    # Options to be added:
                    instruction += " --dataset " + str(d)
                    instruction += " --interface " + i
                    instruction += " --C " + str(c)
                    instruction += " --rins " + str(r)
                    #instruction += " --polishtime " + str(p_time)

                    # The following three instructions are needed to construct the folder's name

                    instruction += " --outfolder " + current_folder
                    # Setting the timeout and saving the output to a log file:
                    instruction += " --timeout 600 \n"

                    fp.write(instruction)


print("Copying files")
for file in files:
    file_remote = remote_path + file
    file_local = local_path + file

    print(file_remote + ' >>> ' + file_local)
    try:
        sftp.remove(file_remote)
    except IOError:
        print("File was not on the cluster")

    sftp.put(file_local, file_remote)

sftp.close()


ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("export SGE_ROOT=/usr/share/gridengine")

print(ssh_stdout.read())
print(ssh_stderr.read())


print("qsub {0}scripts/commands.job -cwd".format(remote_path))
# Give this job to the cluster
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("qsub {0}scripts/commands.job -cwd".format(remote_path))


print(ssh_stdout.read())
print(ssh_stderr.read())
os.remove("commands.job")
ssh.close()
