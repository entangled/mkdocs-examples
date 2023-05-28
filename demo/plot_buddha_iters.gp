# ~/~ begin <<docs/buddhabrot.md#demo/plot_buddha_iters.gp>>[init]
set term svg size 1000 400
# ~/~ begin <<docs/buddhabrot.md#blue-red-palette>>[init]
rcol(x) = 0.237 - 2.13*x + 26.92*x**2 - 65.5*x**3 + 63.5*x**4 - 22.36*x**5
gcol(x) = ((0.572 + 1.524*x - 1.811*x**2)/(1 - 0.291*x + 0.1574*x**2))**2
bcol(x) = 1/(1.579 - 4.03*x + 12.92*x**2 - 31.4*x**3 + 48.6*x**4 - 23.36*x**5)
set palette model RGB functions rcol(gray), gcol(gray), bcol(gray)
# ~/~ end
set size ratio -1
set xlabel "Re"
set ylabel "Im"
# set log cb
set xrange [-1.1:-0.1]
set yrange [-0.3:0.7]
unset key; unset colorbox
set bmargin 5
set lmargin 5
set multiplot layout 1, 3
set cbrange [1:25]
set title "100 iterations"
plot 'data/buddha0000100.dat' matrix nonuniform u 1:2:($3+1) w image
unset ytics; unset ylabel
set cbrange [1:50]
set title "10,000 iterations"
plot 'data/buddha0010000.dat' matrix nonuniform u 1:2:($3+1) w image
set cbrange [1:100]
set title "1,000,000 iterations"
plot 'data/buddha1000000.dat' matrix nonuniform u 1:2:($3+1) w image
unset multiplot
# ~/~ end