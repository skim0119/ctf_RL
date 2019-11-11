import tensorflow as tf
from os import walk
import csv

def GetFileContents(path):
    p_attack =[];p_def=[];p_scout=[]
    r_attack =[];r_def=[];r_scout=[]
    wr =[];r=[]
    for e in tf.train.summary_iterator(path):
        for v in e.summary.value:
            if v.tag == 'adapt_train_log/perc_attack':
                p_attack.append(v.simple_value)
            elif v.tag == 'adapt_train_log/perc_defense':
                p_def.append(v.simple_value)
            elif v.tag == 'adapt_train_log/perc_scout':
                p_scout.append(v.simple_value)
            elif v.tag == 'adapt_train_log/reward':
                r.append(v.simple_value)
            elif v.tag == 'adapt_train_log/reward_defense':
                r_def.append(v.simple_value)
            elif v.tag == 'adapt_train_log/reward_scout':
                r_scout.append(v.simple_value)
            elif v.tag == 'adapt_train_log/reward_attack':
                r_attack.append(v.simple_value)
            elif v.tag == 'adapt_train_log/win-rate':
                wr.append(v.simple_value)

    return p_attack,p_def,p_scout,r_attack,r_def,r_scout,r,wr


def WriteToCSV(trialName,p_attack,p_def,p_scout,r_attack,r_def,r_scout,r,wr):
    with open(trialName+'.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for i in range(len(p_attack)):
            csv_writer.writerow([p_attack[i],p_def[i],p_scout[i],r_attack[i],r_def[i],r_scout[i],r[i],wr[i]])

# 'adapt_train_log/perc_attack'
# 'adapt_train_log/perc_defense'
# 'adapt_train_log/perc_scout'
# 'adapt_train_log/reward'
# 'adapt_train_log/reward_defense'
# 'adapt_train_log/reward_scout'
# 'adapt_train_log/reward_attack'
# 'adapt_train_log/win-rate'

if __name__ == "__main__":
    # path_to_events_file = "/home/neale/TRIAL_v2/logs/TRIAL3_CONFID_BASELINE_N_TARGET_SD_1571398073.6999316/events.out.tfevents.1571398080.ccc0151"
    mypath = "/home/neale/TRIAL_v2/logs/"
    for (dirpath, dirnames, filenames) in walk(mypath):
        if len(filenames) != 0:
            trialName = dirpath.split("/")[-1]
            print(trialName)
            p_attack,p_def,p_scout,r_attack,r_def,r_scout,r,wr = GetFileContents(dirpath+"/"+filenames[0])
            WriteToCSV(trialName,p_attack,p_def,p_scout,r_attack,r_def,r_scout,r,wr)
