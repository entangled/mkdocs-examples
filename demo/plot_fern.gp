# ~/~ begin <<docs/l-systems.md#demo/plot_fern.gp>>[init]
set term svg size 820 1000
load 'demo/preamble.gp'
set xrange [-170:100]
set yrange [-10:340]
plot '< python -c "from demo.lsystem import barnsley_fern; barnsley_fern.to_gnuplot(7)"' \
     u 2:1 w l t'' ls 1
# ~/~ end