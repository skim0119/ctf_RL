# python ppo_subpolicy_confid.py 09_08_19_THRESH_P1_01_P3_01 1 1 0.01 0.01 &
# python ppo_subpolicy_confid.py 09_08_19_THRESH_P1_05_P3_02 1 1 0.05 0.02 &
# python ppo_subpolicy_confid.py 09_08_19_THRESH_P1_01_P3_01 1 1 0.01 0.01 &
# python ppo_subpolicy_confid.py 09_08_19_THRESH_P1_05_P3_02 1 1 0.05 0.02 &

# python ppo_subpolicy_confid.py 09_08_19_CONFID_P1_01_P2_15 2 1 0.01 0.15 &
# python ppo_subpolicy_confid.py 09_08_19_CONFID_P1_02_P2_15 2 1 0.05 0.15 &
# python ppo_subpolicy_confid.py 09_08_19_CONFID_P1_03_P2_15 2 1 0.01 0.15 &
# python ppo_subpolicy_confid.py 09_08_19_CONFID_P1_04_P2_15 2 1 0.05 0.15 &

# python ppo_subpolicy_confid.py 09_07_19_FS_P1_2 3 1 2 0 &
# python ppo_subpolicy_confid.py 09_07_19_FS_P1_3 3 1 3 0 &
# python ppo_subpolicy_confid.py 09_07_19_FS_P1_4 3 1 4 0 &
# python ppo_subpolicy_confid.py 09_07_19_FS_P1_5 3 1 5 0 &


python ppo_subpolicy_confid_target.py f2.ini F2_CONFID1_SUB_1 2 1 0.01 0.10 0 /device:GPU:1 &
python ppo_subpolicy_confid_target.py f2.ini F2_CONFID1_SUB_2 2 1 0.02 0.10 0 /device:GPU:1 &
python ppo_subpolicy_confid_target.py s2.ini S2_CONFID1_SUB_1 2 1 0.01 0.10 0 /device:GPU:1 &
python ppo_subpolicy_confid_target.py s2.ini S2_CONFID1_SUB_2 2 1 0.02 0.10 0 /device:GPU:1 &
python ppo_subpolicy_confid_target.py f2.ini F2_CONFID1_NSUB_1 2 0 0.01 0.10 0 /device:GPU:0 &
python ppo_subpolicy_confid_target.py f2.ini F2_CONFID1_NSUB_2 2 0 0.02 0.10 0 /device:GPU:0 &
python ppo_subpolicy_confid_target.py s2.ini S2_CONFID1_NSUB_1 2 0 0.01 0.10 0 /device:GPU:0 &
python ppo_subpolicy_confid_target.py s2.ini S2_CONFID1_NSUB_2 2 0 0.02 0.10 0 /device:GPU:0 &
