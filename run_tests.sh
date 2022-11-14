echo "Running unit tests..."
python -m unittest

echo ""
echo "Running SIAM xp 1 test"
cd experiments/SIAM/xp_1_balls
python xp_a_accuracy_sol.py --id Test --noverbose
python xp_b_screening.py --id Test --noverbose
python xp_c_viz.py  --id Test --noverbose --noshow
cd ..

echo ""
echo "Running SIAM xp 2 test"
cd xp_2_bench_time
python xp_a_get_state.py --id Test --precision 4 --noverbose
python xp_b_get_budget.py --id Test --precision 4 --noverbose
python xp_c_results_time.py --id Test --precision 4 --noverbose
python xp_d_viz.py --id Test --precision 4 --noverbose --noshow
cd ../../../
echo "End of testing"