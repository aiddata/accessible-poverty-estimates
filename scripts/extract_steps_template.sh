
username=smgoodman
output_name=GH_2014_DHS
data_dir=/home/userx/Desktop/accessible-poverty-estimates/data


# -----------------
# STEP 1: copy files from local to sciclone


ssh $username@bora.sciclone.wm.edu "mkdir -p ape_extracts/$output_name"
scp -r $data_dir/outputs/$output_name/*json smgoodman@bora.sciclone.wm.edu:ape_extracts/$output_name


# -----------------
# STEP 2: SSH to SciClone

ssh $username@bora.sciclone.wm.edu

# -----------------
# STEP 3: start interactive job to run extract builder

qsub -I -l nodes=1:c18c:ppn=12 -l walltime=1:00:00


# -----------------
# STEP 4: run extract builder

cd ~/ape_extracts/$output_name

python /sciclone/aiddata10/geo/master/source/geo-hpc/extract-scripts/builder.py extract_job.json

# exit from ssh connection
exit

# -----------------
# STEP 5: copy files from sciclone to local (when extract has completed)

scp smgoodman@bora.sciclone.wm.edu:ape_extracts/$output_name/ioe/*/*/merge*.csv $data_dir/outputs/$output_name
