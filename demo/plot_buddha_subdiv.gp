# ~/~ begin <<docs/buddhabrot.md#demo/plot_buddha_subdiv.gp>>[init]
set term svg size 1000, 500
# ~/~ begin <<docs/buddhabrot.md#blue-red-palette>>[init]
rcol(x) = 0.237 - 2.13*x + 26.92*x**2 - 65.5*x**3 + 63.5*x**4 - 22.36*x**5
gcol(x) = ((0.572 + 1.524*x - 1.811*x**2)/(1 - 0.291*x + 0.1574*x**2))**2
bcol(x) = 1/(1.579 - 4.03*x + 12.92*x**2 - 31.4*x**3 + 48.6*x**4 - 23.36*x**5)
set palette model RGB functions rcol(gray), gcol(gray), bcol(gray)
# ~/~ end
unset key
unset colorbox
set xlabel "Re"
set ylabel "Im"
set xrange [-0.5:0.5]
set yrange [0.2:1.2]

set multiplot layout 1, 2
set title "subdivisions"
plot 'data/buddha_precom.dat' matrix nonuniform u 1:2:(int($3) & 0x3C) w image
set title "classification"
plot 'data/buddha_precom.dat' matrix nonuniform u 1:2:(int($3) & 0xC0) w image
unset multiplot
# ~/~ end