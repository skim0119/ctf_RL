python run_multiagent_ppo.py --train_number 00 --machine TL --map_size 30 --nbg 3 --nba 0 --nrg 3 --nra 0 &
python run_cvdc.py --train_number 00 --machine TL --map_size 30 --nbg 3 --nba 0 --nrg 3 --nra 0
python run_multiagent_ppo.py --train_number 01 --machine TL --map_size 30 --nbg 2 --nba 1 --nrg 3 --nra 0 &
python run_cvdc.py --train_number 01 --machine TL --map_size 30 --nbg 2 --nba 1 --nrg 3 --nra 0
