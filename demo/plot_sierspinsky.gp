# ~/~ begin <<docs/l-systems.md#demo/plot_sierspinsky.gp>>[init]
set term svg size 1000 580
load 'demo/preamble.gp'
set multiplot layout 2, 3
set xrange [-0.1:1.1]; set yrange [-0.1:0.95]
plot '< python -c "from demo.lsystem import sierspinsky; sierspinsky.to_gnuplot(0)"' \
     u 1:2 w l t'gen 0' ls 1
set xrange [-0.2:2.2]; set yrange [-0.2:1.9]
plot '< python -c "from demo.lsystem import sierspinsky; sierspinsky.to_gnuplot(1)"' \
     u 1:2 w l t'gen 1' ls 1
set xrange [-0.4:4.4]; set yrange [-0.4:3.8]
plot '< python -c "from demo.lsystem import sierspinsky; sierspinsky.to_gnuplot(2)"' \
     u 1:2 w l t'gen 2' ls 1
set xrange [-0.8:8.8]; set yrange [-0.8:7.6]
plot '< python -c "from demo.lsystem import sierspinsky; sierspinsky.to_gnuplot(3)"' \
     u 1:2 w l t'gen 3' ls 1
set xrange [-1.6:17.6]; set yrange [-1.6:15.2]
plot '< python -c "from demo.lsystem import sierspinsky; sierspinsky.to_gnuplot(4)"' \
     u 1:2 w l t'gen 4' ls 1
set xrange [-3.2:35.2]; set yrange [-3.2:30.4]
plot '< python -c "from demo.lsystem import sierspinsky; sierspinsky.to_gnuplot(5)"' \
     u 1:2 w l t'gen 5' ls 1
unset multiplot
# ~/~ end