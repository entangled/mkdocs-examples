# ~/~ begin <<docs/l-systems.md#demo/plot_koch.gp>>[init]
set term svg size 820 1000
load 'demo/preamble.gp'
set multiplot layout 4, 1
set xrange [-50:2250]
set yrange [-50:670]
plot '< python -c "from demo.lsystem import koch; koch(60).to_gnuplot(7)"' \
     u 1:2 w l t'' ls 1
set xrange [-30:870]
set yrange [-30:330]
plot '< python -c "from demo.lsystem import koch; koch(72).to_gnuplot(7)"' \
     u 1:2 w l t'' ls 1
set xrange [-10:400]
set yrange [-10:180]
plot '< python -c "from demo.lsystem import koch; koch(80).to_gnuplot(7)"' \
     u 1:2 w l t'' ls 1
set xrange [-7:160]
set yrange [-7:80]
plot '< python -c "from demo.lsystem import koch; koch(88.5).to_gnuplot(7)"' \
     u 1:2 w l t'' ls 1
unset multiplot
# ~/~ end