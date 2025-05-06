

for tag in 'mb' 'rl'
do
    for additional in '9' '11' '15'
    do
        python3 plot_traj_hull.py $tag $additional
    done
done