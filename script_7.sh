python run_multiagent_ppo.py --train_number 00 --machine TL --map_size 30 --nbg 7 --nba 0 --nrg 7 --nra 0 &
python run_cvdc.py --train_number 00 --machine TL --map_size 30 --nbg 7 --nba 0 --nrg 7 --nra 0
python run_multiagent_ppo.py --train_number 01 --machine TL --map_size 30 --nbg 6 --nba 1 --nrg 7 --nra 0 &
python run_cvdc.py --train_number 01 --machine TL --map_size 30 --nbg 6 --nba 1 --nrg 7 --nra 0
python run_multiagent_ppo.py --train_number 02 --machine TL --map_size 30 --nbg 5 --nba 2 --nrg 7 --nra 0 &
python run_cvdc.py --train_number 02 --machine TL --map_size 30 --nbg 5 --nba 2 --nrg 7 --nra 0
python run_multiagent_ppo.py --train_number 03 --machine TL --map_size 30 --nbg 4 --nba 3 --nrg 7 --nra 0 &
python run_cvdc.py --train_number 03 --machine TL --map_size 30 --nbg 4 --nba 3 --nrg 7 --nra 0
