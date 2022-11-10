echo "Running unit tests..."
python -m unittest

echo ""
echo "Running SIAM xp test 1"
cd experiments/SIAM/xp_1_balls
python xp_a_accuracy_sol.py --id Test --noverbose
python xp_b_screening.py --id Test --noverbose
python xp_c_viz_fig1.py  --id Test --noverbose
python xp_c_viz_fig2.py  --id Test --noverbose