# ~/~ begin <<docs/buddhabrot.md#Makefile>>[init]
.RECIPEPREFIX = >

# ~/~ begin <<docs/l-systems.md#build>>[init]
include/sierspinsky-table.md: demo/sierspinsky_table.py demo/lsystem.py demo/turtle.py
> @mkdir -p $(@D)
> python -m demo.sierspinsky_table > $@
# ~/~ end
# ~/~ begin <<docs/l-systems.md#build>>[1]
docs/fig/sierspinsky.svg: demo/plot_sierspinsky.gp demo/lsystem.py demo/preamble.gp demo/turtle.py
> @mkdir -p $(@D)
> gnuplot $< > $@
# ~/~ end
# ~/~ begin <<docs/l-systems.md#build>>[2]
docs/fig/dragon.svg: demo/plot_dragon.gp demo/lsystem.py demo/preamble.gp
> @mkdir -p $(@D)
> gnuplot $< > $@
# ~/~ end
# ~/~ begin <<docs/l-systems.md#build>>[3]
docs/fig/fern.svg: demo/plot_fern.gp demo/lsystem.py demo/preamble.gp
> @mkdir -p $(@D)
> gnuplot $< > $@
# ~/~ end
# ~/~ begin <<docs/l-systems.md#build>>[4]
docs/fig/koch.svg: demo/plot_koch.gp demo/lsystem.py demo/preamble.gp
> @mkdir -p $(@D)
> gnuplot $< > $@
# ~/~ end
# ~/~ begin <<docs/buddhabrot.md#build>>[0]
cargo_args += --manifest-path=demo/buddhabrot/Cargo.toml
cargo_args += --release

target_path := demo/buddhabrot/target/release
buddhabrot := $(target_path)/buddhabrot

-include $(target_path)/buddhabrot.d
$(buddhabrot):
> cargo build $(cargo_args)

data/buddha%.dat: $(buddhabrot)
> @mkdir -p $(@D)
> $(buddhabrot) -W 1024 -H 1024 -m $(*F) -s 10 --gnuplot $@

docs/fig/buddha_iterations.svg: \
    demo/plot_buddha_iters.gp \
    data/buddha0010000.dat \
    data/buddha0100000.dat \
    data/buddha1000000.dat
> @mkdir -p $(@D)
> gnuplot $< > $@
# ~/~ end
# ~/~ end