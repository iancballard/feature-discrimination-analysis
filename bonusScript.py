# Simple python code for automatically bonusing workers on mTurk.
# Written by Desmond Ong (github.com/desmond-ong), 15 July 2013. Send comments to dco@stanford.edu.

# Instructions:
#   1) replace "filename" with the name of the input file,
#   2) write the bonus message to participants,
#   3) fill in the location where CLT is installed.
# Then
#   4) run "python bonusScript.py"
#   5) run "sh bonusBashScript.sh"

# You may also need to export your javahome. First run:
# /usr/libexec/java_home
# then run, replacing the path on the RHS of the = with the output from the above command
# export JAVA_HOME=/System/Library/Java/JavaVirtualMachines/1.6.0.jdk/Contents/Home
# export MTURK_CMD_HOME=/Applications/aws-mturk-clt-1.3.1

# Format for filename: a csv file with:
# AssignmentID (*not* HIT ID) in the first column,
# workerID in the second column,
# and bonus amount in the third column (in dollars, no dollar sign).
# E.g.
#
# AssignmentID   WorkerID    Bonus
# 2XXX11X1X1X1XXXXX1XXXXX1XXXXXX  X12XXXZXXXX73X  0.5
# 2XXX1XXXXX1XXXXXX1X1XXXXXXXXXX  X13XXX4X5XXX6X  0.27
#
#edited by ian to work with text instead of csv files

import os
data_dir = '/Users/ianballard/Dropbox/fd/results/'
file_num = input('Enter payment file number: ')
filename = os.path.join(data_dir,'payment' + str(file_num) + '.txt')
f=open(filename,'r')

bonusMessage = "Your bonus for performing my hit. My deepest apologies for being so late. Some unavoidable conflicts came up. I added $1.50 to your hit as compensation."
locationofCLT = "/Applications/aws-mturk-clt-1.3.1"

rowNum = 0
bonusScripts = ""

for line in f.readlines():
	line = line.strip('\n')
	line = line.split('\t')
	if float(line[2]) > 0:  # if bonus greater than 0
		bonusScripts = bonusScripts + "./grantBonus.sh -workerid " + line[1] + " -amount " + line[2] + " -assignment " + line[0] + " -reason " + "\"" + bonusMessage + "\" \n"
	rowNum += 1
        

#write the bash script for running the bonus commands
bonusBashScript = open("bonusBashScript.sh", 'w')
bonusBashScript.write("#!/usr/bin/env sh\npushd " + locationofCLT + "/bin\n" + bonusScripts + "popd")
bonusBashScript.close()
