#! /usr/bin/gnuplot --persist
#####################################################################
#
# Plot the RoboND-DeepRL parameter tuning results
#				Douglas Teeple June 2018
#
# Columns:
#         1           2        3          4             5             6
# SuccessfulGrabs TotalRuns Accuracy LearningRate maxLearningRate LSTMSize
#
#####################################################################
firstrow = system('head -1 gazebo-arm.plt')
set xlabel word(firstrow, 2)
set ylabel word(firstrow, 3)
LearnRate = word(firstrow, 7)

set yrange [0:1]
#set xrange [0:500]
set nokey

set title "Robo-ND DeepRL LSTM Size and Learning Rate Plots\nDouglas Teeple June 2018" font "Bold-Times-Roman,12"
set macros
POS  = "at graph 0.2,0.9 font ',8'"
POS2 = "at graph 0.2,0.85 font ',8'"
SKIPPER = "every ::1 using 2:3 with lines"
stats 'gazebo-arm.plt' using 3 nooutput
set label 1 'LSTM-512 LearnRate '.LearnRate @POS
set label 2 sprintf("Max Acc: %0.2f Run: %d\n(after settling)", STATS_max, STATS_index_max) @POS2
set arrow 1 from STATS_index_max,STATS_max+0.1 to STATS_index_max,STATS_max linecolor rgb "red"
plot 'gazebo-arm.plt' @SKIPPER


