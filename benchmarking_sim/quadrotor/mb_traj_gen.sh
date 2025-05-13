
for traj_len in 9 10 11 12 13 14 15
do
    python3 mb_experiment.py 'ilqr' $traj_len
done

# for traj_len in 9 10 11 12 13 14 15
# do
#     python3 mb_experiment.py 'mpc_acados' $traj_len
# done