# ~/~ begin <<docs/l-systems.md#demo/plot_dragon.gp>>[init]
set term svg size 1000 820
load 'demo/preamble.gp'
set xrange [-53:28]
set yrange [-14:52]
plot '< python -c "from demo.lsystem import dragon; dragon.to_gnuplot(11)"' \
     u 1:2 w l t'' ls 1
# ~/~ end