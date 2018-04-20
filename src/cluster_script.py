#!/usr/bin/env python3
import os
import time
import datetime
import paramiko
import getpass

'''
    --- WARNING ---
    FIRST TIME INFO, NECESSARY TODOs
    
    (1)     Se non hai mai avviato questo script, come prima cosa esegui il login:
            >>> ssh username@login.dei.unipd.it
            >>> git clone <url>
            
            (sì, la repository nel nostro spazio dei è necessaria)
            tl; dr: clonare la repository dentro la 'home' del nostro spazio sul DEI.
    
    (2)     Se non l'hai già fatto, scarica networkx dal sito (zip) ed estrai la cartella nella tua home.
'''

# Parametri per l'esecuzione

# input files
proteinsin = "../data/hint+hi2012_index_file.txt"
samplesin = "../data/snvs.tsv"
genesin = "../data/hint+hi2012_edge_file.txt"
prob = [True, False]  # Probabilistic version of the problem or not?
ks = [2,3,4,5,6,7,8,9,10]  # On which ks do we want to test our algorithm?
delta = 0.8  # for now, delta doesn't really matter to the analysis
time_out = 604800  # for now, for each execution, we're willing to wait 7 days per run, maximum

server = "login.dei.unipd.it"

# Crea un'istanza del client SSH
ssh = paramiko.SSHClient()
# <<Required, since "login.dei.unipd.it" is not a "well known ssh host">> (???)
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Per evitare di scrivere il proprio username, se la macchina dalla quale lo sia avvia è "nota"
if getpass.getuser() == 'venir':
    pwd = str(getpass.getpass(prompt="Password: "))
    username = "venirluca"
elif getpass.getuser() == 'iltuonomesultuoPC':
    pwd = str(getpass.getpass(prompt="Password: "))
    username = "iltuonomeDEI"
else:
    username = str(input("Please, enter your username (@" + server + "): "))
    pwd = str(getpass.getpass(prompt=username+"@" + server + ": "))

# Connesione al server
ssh.connect(server, username=username, password=pwd)

# Apertura di un'istanza per il file transfer (con secure file transfer protocol)
sftp = ssh.open_sftp()  # per trasferire i file

# Cartella remota del nostro progetto
remote_path = "/home/" + username + "/algobio1718/"

# Cartella locale del nostro progetto
local_path = os.path.dirname(os.getcwd()) + "/"

# Main output folder:
current_time = time.time()
timestamp = datetime.datetime.fromtimestamp(current_time).strftime('%Y%m%d_%H:%M:%S')
current_folder = "run__" + timestamp

# È necessario creare la cartella /out/ in remoto!
try:
    sftp.mkdir(remote_path + "out/")
except IOError:
    # Se è già stata creata, non occorre ri-crearla
    pass

# Dato che la cartella corrente è un timestamp, siamo sicuri di poterla creare sempre (in remoto)
sftp.mkdir(remote_path + "out/" + current_folder)

# Files to be uploaded
files = ['src/main.py', 'src/lib/core.py', 'src/lib/inout.py']

# Create a local file that will be sent to the server (the infamous '.job' file)
with open("commands.job", "w") as fp:
    fp.write("#!/bin/bash \n")
    #fp.write("export PYTHONPATH=$PYTHONPATH:/nfsd/opt/CPLEX12.6/cplex/python/3.4/x86-64_linux/ \n")
    fp.write("export PYTHONPATH=$PYTHONPATH:$HOME/networkx/ \n")
    for k in ks:
        for p in probs:

            # Formatting/constructing the instruction to be given:
            instruction = "python3 "+ remote_path + "src/main.py"

            # Options to be added:
            instruction += " --k " + str(k)
            instruction += " --delta " + str(delta)
            if p:
                instruction += " --prob "
            instruction += " --proteinsin " + proteinsin
            instruction += " --samplesin " + samplesin
            instruction += " --genesin " + genesin

            # current outputfolder for now is not used in our code
            # instruction += " --outfolder " + output_folder

            # Timeout is not currently used in our code
            # instruction += " --timeout 600 " + str(time_out)

            # Saving the output to a log file:
            output_logfilename = 'k='+str(k) + '_' + 'delta='+str(d)
            instruction += ' > ' + remote_path + "out/" + current_folder +'/'+ output_logfilename + '_results.log'
            instruction += '\n'
            fp.write(instruction)

print("Copying files")
for file in files:
    file_remote = remote_path + file
    file_local = local_path + file

    print(file_local + ' >>> ' + file_remote)
    try:
        sftp.remove(file_remote)
    except IOError:
        print(file + " was not on the cluster")
        pass

    sftp.put(file_local, file_remote)

# Put the file on the current folder on the cluster and delete the local one
print(local_path + 'src/commands.job' ' >>> ' + remote_path + 'out/' + current_folder + '/commands.job')
sftp.put(local_path + 'src/commands.job', remote_path + 'out/' + current_folder + '/commands.job')
os.remove("commands.job")

# Give this job to the cluster
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("export SGE_ROOT=/usr/share/gridengine \n" +  #  necessary
                                                     "cd {0}out/{1} \n".format(remote_path, current_folder) +  # also necessary
                                                     "qsub -cwd commands.job")
# dev'essere "tutto assieme"
# o si dimentica dell'export.
# una singola chiamata di exec_command fa dimenticare tutto quello che è stato fatto prima
# qsub -cwd == "current working directory". DEVE ESSERE MESSO PRIMA!!!

# Print output and errors
print(ssh_stdout.read().decode('utf-8'))
print(ssh_stderr.read().decode('utf-8'))

sftp.close()
ssh.close()
