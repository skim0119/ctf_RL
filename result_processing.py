import csv
import numpy as np
from os import walk

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def ProcessCTF(fileName,baseline=True):
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        wr = []
        for row in csv_reader:
            wr.append(float(row[7]))

    ss = np.average(wr[-int(0.2*float(len(wr))):-1])
    if not baseline:
        pDrop = np.average(np.array(wr[0:2]))
        counter = 0
        for i,w in enumerate(wr):
            if w > 0.95*ss and w < 1.05*ss:
                counter += 1
            else:
                counter = 0
            if counter==3:
                return ss , pDrop , i
        return ss , pDrop , 0
    return ss

def PlotCTF_1(fileName,figure,smoothing,f,c):
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        wr = []
        for row in csv_reader:
            wr.append(float(row[7]))
    wr = smooth(wr,smoothing)
    if "FIXED" in fileName:
        if f==0:
            figure.plot(np.arange(0, len(wr)),wr, "r",label='Fixed Step')
            f=1
        else:
            figure.plot(np.arange(0, len(wr)),wr, "r")
    else:
        if c==0:
            figure.plot(np.arange(0, len(wr)),wr, "b",label='Confidence')
            c=1
        else:
            figure.plot(np.arange(0, len(wr)),wr, "b")
    return f,c

def PlotCTF_2(fileName,figure,smoothing):
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        pa = [];pd = [];ps = [];wr=[]
        for row in csv_reader:
            pa.append(float(row[0]))
            pd.append(float(row[1]))
            ps.append(float(row[2]))
            wr.append(float(row[7]))
    ln1 = np.array(pa)*100
    ln2 = (np.array(pa) + np.array(pd))*100
    ln3 = (np.array(pa) + np.array(pd)+ np.array(ps))*100

    figure.fill_between(np.arange(0, len(pa)),0,ln1,color="r",label="Attack")
    figure.fill_between(np.arange(0, len(pa)),ln1,ln2,color="b",label="Defense")
    figure.fill_between(np.arange(0, len(pa)),ln2,100,color="black",label="Scout")
    # figure.plot(np.arange(0, len(pa)),ln1, "b")
    # figure.plot(np.arange(0, len(pa)),ln2, "b")
    # figure.plot(np.arange(0, len(pa)),ln3, "b")
    ss = np.average(wr[-int(0.2*float(len(wr))):-1])
    return ss

def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

if __name__ == "__main__":
    if 1:
        mypath = "/media/neale/My Passport/Baseline Results/"
        for (dirpath, dirnames, filenames) in walk(mypath):
            if len(filenames) != 0:
                for fileName in filenames:
                    trialName = fileName
                    ss = ProcessCTF(dirpath+"/"+fileName)
                    print(trialName,ss)

    if 1:
        mypath = "/media/neale/My Passport/Adaptation Results/"
        for (dirpath, dirnames, filenames) in walk(mypath):
            if len(filenames) != 0:
                for fileName in filenames:
                    trialName = fileName
                    ss,pDrop,riseTime = ProcessCTF(dirpath+"/"+fileName,False)
                    print(trialName,ss,pDrop,riseTime)

    if 0: #Figure of Failure of Fixed Step.
        mypath = "/media/neale/My Passport/Adaptation Results/"
        fig = plt.figure()
        fig.set_size_inches(12.5, 5.5)
        ax1 = fig.add_subplot(111)
        ax1.set_title('Confidence vs. Fixed Step Hierarchy')
        ax1.set_xlabel('Episodes [$10^3$]')
        ax1.set_ylabel('Win Rate [%]')
        f=0;c=0
        for (dirpath, dirnames, filenames) in walk(mypath):
            if len(filenames) != 0:
                for fileName in filenames:
                    trialName = fileName
                    if "SD" in trialName:
                        f,c=PlotCTF_1(dirpath+"/"+fileName, ax1,0.9,f,c)

        ax1.legend()
        plt.show()


    if 0: #Figure for Transience
        mypath = "/media/neale/My Passport/Adaptation Results/"
        fig = plt.figure()
        fig.set_size_inches(9.5, 7.5)
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        fig.suptitle('Primative Percentages')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        ax.set_xlabel('Episodes [$10^3$]')
        ax.set_ylabel('Primative Network Percentage')
        ax1.yaxis.set_major_formatter(PercentFormatter())
        ax2.yaxis.set_major_formatter(PercentFormatter())
        ax1.set_xlim(0,140)
        ax2.set_xlim(0,140)


        fileName = "TRIAL1_CONFID_BASELINE_N_TARGET_SD_1571404626.3464842.csv"
        wr = PlotCTF_2(mypath+fileName, ax1,0.98)
        ax1.set_title('Confidence Based Hierarchy - Win-Rate=' + str(round(wr,3)) )
        fileName = "TRIAL1_FIXED_BASELINE_N_TARGET_SD_1571394349.0140479.csv"
        wr = PlotCTF_2(mypath+fileName, ax2,0.98)
        ax2.set_title('Fixed Step Based Hierarchy - Win-Rate='+ str(round(wr,3))  )
        ax1.legend()
        ax2.legend()
        plt.show()

    if 0: #Figure for Experiments Slide
        mypath = "/media/neale/My Passport/Adaptation Results/"
        fig = plt.figure()
        fig.set_size_inches(12.5, 5.5)
        plt.rc('legend', fontsize=16)
        plt.rc('axes', titlesize=16)     # fontsize of the axes title
        plt.rc('axes', labelsize=14)
        ax1 = fig.add_subplot(111)
        ax1.set_title('')
        ax1.set_xlabel('Episodes [$10^3$]')
        ax1.set_ylabel('Win-Rate [%]')
        f=0;c=0
        fileName = "TRIAL3_CONFID_BASELINE_S_TARGET_SF_1571398145.4904969.csv"
        # fileName = "TRIAL3_CONFID_BASELINE_S_TARGET_WF_1571398145.4892404.csv"
        # fileName = "TRIAL4_FIXED_BASELINE_N_TARGET_WD_1571400368.189333.csv"
        f,c=PlotCTF_1(mypath+"/"+fileName, ax1,0.96,1,1)
        ax1.axvline(10, color="black",label="Adaptation Event",linestyle='-.')
        ax1.legend()
        plt.show()
