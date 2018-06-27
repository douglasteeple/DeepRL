#! /bin/sh
gnuplot --persist <<EOF
#################################################################################################
#
# Plot the RoboND-DeepRL parameter tuning results
#				Douglas Teeple June 2018
#
# Columns:
#         1           2        3          4             5             6		    7            8     
# SuccessfulGrabs TotalRuns Accuracy LearningRate maxLearningRate LSTMSize Last100Accuracy  Arm
#
#################################################################################################
fname = 'gazebo-arm.plt'
firstrow = system("head -1 ".fname)
set xlabel word(firstrow, 2)
set ylabel word(firstrow, 3)
secondrow = system("awk 'FNR==2{print \$0}' ".fname)
LearnRate = word(secondrow, 4)
LSTMSize = word(secondrow, 6)
Arm = word(secondrow, 8)
today = system("date +'%B %d %Y'")
set yrange [0:1]
set xtics
set ytics 0,0.1,1
set grid

set object rectangle from screen 0,0 to screen 1,1 behind fillcolor '#FDF5E6' fillstyle solid noborder
set title "Robo-ND Deep Q Accuracy Plot\nDouglas Teeple ".today font "Bold-Times-Roman,14"

# How many records for each arm?
count1 = system(sprintf("awk '\$8 == 1' %s | wc -l",fname))
count2 = system(sprintf("awk '\$8 == 2' %s | wc -l",fname))

# Arm 1 stats
stats "< awk '\$8 == 1' ".fname every ::(count1 <= 100?1:101) using (count1 <= 100 ? \$3 : \$7) nooutput
arm1IndexMax = STATS_index_max + (count2 <= 100?0:100)
arm1Max = STATS_max

set label 1 sprintf("LSTM-%s LearnRate %s\n$*", LSTMSize, LearnRate) at graph 0.2,0.9 font 'Bold-Times-Roman,10'
set label 2 sprintf("%0.2f", arm1Max) at arm1IndexMax, arm1Max+0.1 tc lt 2 font 'Bold-Times-Roman,10'
set arrow 1 from arm1IndexMax, arm1Max+0.1 to arm1IndexMax, arm1Max linecolor rgb "red"

# Arm 2 stats
stats "< awk '\$8 == 2' ".fname every ::(count2 <= 100?1:101) using (count2 <= 100 ? \$3 : \$7) nooutput
arm2IndexMax = STATS_index_max + (count2 <= 100?0:100)
arm2Max = STATS_max
set label 3 sprintf("%0.2f", arm2Max) at arm2IndexMax, arm2Max+0.1 tc lt 4 font 'Bold-Times-Roman,10'
set arrow 2 from arm2IndexMax, arm2Max+0.1 to arm2IndexMax, arm2Max linecolor rgb "red"

plot "< awk '\$8 == 1' ".fname using 2:3 with lines title "Arm 1",\
	 "< awk '\$8 == 1' ".fname every ::(count1 <= 100?1:101) using 2:7 with lines title "<100>", \
	 "< awk '\$8 == 2' ".fname using 2:3 with lines title "Arm 2",\
	 "< awk '\$8 == 2' ".fname every ::(count2 <= 100?1:101) using 2:7 with lines title "<100>"
EOF

