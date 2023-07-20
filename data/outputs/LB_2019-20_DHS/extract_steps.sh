
# -----------------
# STEP 1: copy files from local to sciclone


ssh -J smgoodman@stats.wm.edu smgoodman@bora.sciclone.wm.edu "mkdir -p ape_extracts/LB_2019-20_DHS"
scp -o 'ProxyJump smgoodman@stat.wm.edu' -r /home/userx/Desktop/accessible-poverty-estimates/data/outputs/LB_2019-20_DHS/*json smgoodman@bora.sciclone.wm.edu:ape_extracts/LB_2019-20_DHS


# -----------------
# STEP 2: SSH to SciClone

#

# -----------------
# STEP 3: start interactive job to run extract builder

qsub -I -l nodes=1:c18c:ppn=1 -l walltime=1:0:0


# -----------------
# STEP 4: run extract builder

cd ~/ape_extracts/LB_2019-20_DHS

python /sciclone/aiddata10/geo/master/source/geo-hpc/extract-scripts/builder.py extract_job.json

exit

# -----------------
# STEP 5: copy files from sciclone to local (when extract has completed)

scp -P 2222 ~/ape_extracts/LB_2019-20_DHS/ioe/*/*/merge*.csv userx@68.0.46.157:/home/userx/Desktop/accessible-poverty-estimates/data/outputs/LB_2019-20_DHS
