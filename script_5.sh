python run_multiagent_ppo.py --train_number 00 --machine TL --map_size 30 --nbg 5 --nba 0 --nrg 5 --nra 0 &
python run_cvdc.py --train_number 00 --machine TL --map_size 30 --nbg 5 --nba 0 --nrg 5 --nra 0
python run_multiagent_ppo.py --train_number 01 --machine TL --map_size 30 --nbg 4 --nba 1 --nrg 5 --nra 0 &
python run_cvdc.py --train_number 01 --machine TL --map_size 30 --nbg 4 --nba 1 --nrg 5 --nra 0
python run_multiagent_ppo.py --train_number 02 --machine TL --map_size 30 --nbg 3 --nba 2 --nrg 5 --nra 0 &
python run_cvdc.py --train_number 02 --machine TL --map_size 30 --nbg 3 --nba 2 --nrg 5 --nra 0
