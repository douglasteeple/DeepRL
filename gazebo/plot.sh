#! /usr/bin/gnuplot --persist
####################################################################################
#
# Plot the RoboND-DeepRL parameter tuning results
#				Douglas Teeple June 2018
#
# Columns:
#         1           2        3          4             5             6
# SuccessfulGrabs TotalRuns Accuracy LearningRate maxLearningRate Last100Accuracy
#
####################################################################################
fname = 'gazebo-arm.plt'
firstrow = system("head -1 ".fname)
set xlabel word(firstrow, 2)
set ylabel word(firstrow, 3)
secondrow = system("awk 'FNR==2{print \$0}' ".fname)
LearnRate = word(secondrow, 4)
lstmsize = word(secondrow, 6)
set yrange [0:1]
set xrange [0:100]

set key box samplen 1 spacing .5 font ",5" width -0.5

set multiplot layout 2,2 rowsfirst title "Robo-ND Deep Q RL LSTM Size and Learning Rate Plots\nDouglas Teeple June 2018" font "Bold-Times-Roman,12"
set macros
POS  = "at graph 0.2,0.9 font ',8'"
POS2 = "at graph 0.2,0.8 font ',8'"
SKIPPER = "every ::1 using 2:3 title ".LearnRate." with lines"

do for [n in "64 128 256 512"] {
	fname = 'gazebo-arm-'.n.'.plt'
	stats fname using 3 every ::50 nooutput
	set label 1 'LSTM-'.n @POS
	set label 2 sprintf("Max Acc: %0.2f Run: %d\n(after settling)", STATS_max, STATS_index_max) @POS2
	set arrow 1 from STATS_index_max,STATS_max+0.1 to STATS_index_max,STATS_max linecolor rgb "red"
	plot for [i=0:*] fname index i @SKIPPER
}

unset multiplot

