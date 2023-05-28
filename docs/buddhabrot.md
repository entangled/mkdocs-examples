# Buddhabrot Fractals in Rust

You may have heard of the [Mandelbrot fractal](https://en.wikipedia.org/wiki/Mandelbrot_set). It is a nice exercise to render the Mandelbrot fractal using your favourite language. The idea is to iterate the formula

$$z_{n+1} = z_n^2 + c,$$

And figure out how long it takes for this series to diverge. What is even nicer, is to actually plot the orbits $z_n$ for every $c$ and $z_0 = 0$.

This code uses the `ndarray` crate for numeric arrays, `num` for complex numbers, and `rayon` for parallel computing.

``` {.rust #buddha-imports}
use rayon::prelude::*;
use rayon::join;
use indicatif::{ProgressBar, ParallelProgressIterator};
use rand::Rng;
use ndarray::{Array2, indices, s};
use num::{Complex, zero};
use clap::{Parser, ValueEnum};
use std::sync::atomic::{AtomicU8,Ordering};

// use std::error;
use std::fs::File;
use std::io::Write;
```

## TODO

- [ ] compile this code to WASM to create a web demo

## Plotter

The plotter has an interface that lets us call `p.plot(z, 1.0)`, where `z` is a complex number and `1.0` is a weight. The plotter has an array of pixels and info to map the complex plane to the pixels.

``` {.rust #buddha-plotter}
#[derive(Clone)]
struct ComplexImage<T: Sized> {
    pixels: Array2<T>,
    z_min: Complex<f64>,
    res: f64
}

impl<T: Sized> ComplexImage<T> { 
    fn write_matrix<U, F>(&self, filename: &String, mapping: F) -> Result<(), Error> 
    where F: Fn(&T) -> U,
          U: ToString
    {
        let mut file = File::create(filename).map_err(Error::IO)?;
        let (w, _) = self.pixels.dim();
        fn stringify<T>(it: impl Iterator<Item=T>) -> String where T: ToString {
            it.fold(String::new(), |a, b| a + " " + &b.to_string())
        }
        writeln!(file, "{} {}", w, stringify((0..w).map(|i| i as f64 * self.res + self.z_min.re))).map_err(Error::IO)?;
        for (j, row) in self.pixels.columns().into_iter().enumerate() {
            writeln!(file, "{} {}",
                j as f64 * self.res + self.z_min.im,
                stringify(row.iter().map(&mapping))).map_err(Error::IO)?;
        }
        Ok(())
    }
}

struct Cell (AtomicU8);

#[derive(Debug, PartialEq, Eq)]
enum SimplexState {
    Unvisited, Inside, Outside
}

impl SimplexState {
    fn from_u8(x: u8) -> Self {
        match x {
            0b00 => Self::Unvisited,
            0b01 => Self::Unvisited,
            0b10 => Self::Outside,
            0b11 => Self::Inside,
            _    => panic!("invalid number for simplex state")
        }
    }
}

impl std::default::Default for Cell {
    fn default() -> Self {
        Cell(AtomicU8::new(0u8))
    }
}

#[allow(dead_code)]
impl Cell {
    // getters
    fn get(&self) -> u8 { self.0.load(Ordering::Relaxed) }
    fn vertex(&self) -> SimplexState { SimplexState::from_u8(self.get() & 0b00000011 ) }
    fn x_edge(&self) -> SimplexState { SimplexState::from_u8((self.get() >> 2) & 0b00000011 ) }
    fn y_edge(&self) -> SimplexState { SimplexState::from_u8((self.get() >> 4) & 0b00000011 ) }
    fn face(&self) -> SimplexState { SimplexState::from_u8((self.get() >> 6) & 0b00000011 ) }

    // setters
    fn mark_vertex(&self)   { self.0.fetch_or(0b00000011, Ordering::Relaxed); }
    fn mark_x_edge(&self)   { self.0.fetch_or(0b00001100, Ordering::Relaxed); }
    fn mark_y_edge(&self)   { self.0.fetch_or(0b00110000, Ordering::Relaxed); }
    fn mark_face(&self)     { self.0.fetch_or(0b11000000, Ordering::Relaxed); }
    fn unmark_vertex(&self) { self.0.fetch_or(0b00000010, Ordering::Relaxed); }
    fn unmark_x_edge(&self) { self.0.fetch_or(0b00001000, Ordering::Relaxed); }
    fn unmark_y_edge(&self) { self.0.fetch_or(0b00100000, Ordering::Relaxed); }
    fn unmark_face(&self)   { self.0.fetch_or(0b10000000, Ordering::Relaxed); }
}

/// The SampleArea encodes the selection on the CubeComplex.
/// The boundaries are _inclusive_ on both ends, however,
/// due to the structure of the CubeComplex this is a more
/// natural choice.
#[derive(Debug)]
struct SampleArea {
    imin: usize,
    imax: usize,
    jmin: usize,
    jmax: usize
}

impl SampleArea {
    fn size(&self) -> usize { (self.imax - self.imin) * (self.jmax - self.jmin) }
}

impl ComplexImage<Cell> {
    fn horizontal_split(&self, area: &SampleArea, maxit: usize, n: usize) -> (SampleArea, SampleArea) {
        let b = (area.jmin + area.jmax) / 2;
        let y = self.z_min.im + (b as f64) * self.res;
        for a in area.imin..area.imax {
            let x = self.z_min.re + (a as f64) * self.res;
            let mut c = (0..n).map(
                |k| Complex::new(x + (k as f64 / n as f64) * self.res, y));
            if c.all(|c| mandelbrot_test(c, maxit)) {
                self.pixels[(a, b)].mark_x_edge();
            } else {
                self.pixels[(a, b)].unmark_x_edge();
            }
        }
        ( SampleArea { jmax: b, ..*area }, SampleArea { jmin: b, ..*area } )
    }

    fn vertical_split(&self, area: &SampleArea, maxit: usize, n: usize) -> (SampleArea, SampleArea) {
        let a = (area.imin + area.imax) / 2;
        let x = self.z_min.re + (a as f64) * self.res;
        for b in area.jmin..area.jmax {
            let y = self.z_min.im + (b as f64) * self.res;
            let mut c = (0..n).map(
                |k| Complex::new(x, y + (k as f64 / n as f64) * self.res));
            if c.all(|c| mandelbrot_test(c, maxit)) {
                self.pixels[(a, b)].mark_y_edge();
            } else {
                self.pixels[(a, b)].unmark_y_edge();
            }
        }
        ( SampleArea { imax: a, ..*area }, SampleArea { imin: a, ..*area} )
    }

    fn split(&self, area: &SampleArea, maxit: usize, n: usize) -> (SampleArea, SampleArea) {
        if (area.imax - area.imin) > (area.jmax - area.jmin) {
            self.vertical_split(area, maxit, n)
        } else {
            self.horizontal_split(area, maxit, n)
        }
    }

    fn check_border(&self, area: &SampleArea) -> bool {
        let top = self.pixels.slice(s![area.imin..area.imax,area.jmin]);
        let right = self.pixels.slice(s![area.imax,area.jmin..area.jmax]);
        let bottom = self.pixels.slice(s![area.imin..area.imax,area.jmax]);
        let left = self.pixels.slice(s![area.imin,area.jmin..area.jmax]);
        top.iter().all(|c| c.x_edge() == SimplexState::Inside) &&
        right.iter().all(|c| c.y_edge() == SimplexState::Inside) &&
        bottom.iter().all(|c| c.x_edge() == SimplexState::Inside) &&
        left.iter().all(|c| c.y_edge() == SimplexState::Inside)
    }

    fn mark_area(&self, area: &SampleArea) {
        for cell in self.pixels.slice(s![area.imin..area.imax,area.jmin..area.jmax]) {
            cell.mark_face()
        }
    }

    fn step(&self, area: &SampleArea, maxit: usize, n: usize, progress: ProgressBar) {
        if self.check_border(area) {
            self.mark_area(area);
            progress.inc(area.size().try_into().unwrap());
            return
        }

        if area.size() == 1 {
            self.pixels[(area.imin, area.jmin)].unmark_face();
            progress.inc(1);
            return
        }

        if area.size() == 0 {
            return
        }

        let (a, b) = self.split(area, maxit, n);
        join(
            || self.step(&a, maxit, n, progress.clone()), 
            || self.step(&b, maxit, n, progress.clone()));
    }

    fn compute(width: usize, height: usize, z_min: Complex<f64>, res: f64, maxit: usize, subs: usize) -> Self {
        println!("Pre-computing Mandelbrot set.");
        let progress = ProgressBar::new((width * height).try_into().unwrap());
        let obj = ComplexImage { 
            pixels: Array2::default((width+1, height+1)),
            z_min, res
        };
        let area = SampleArea { imin: 0, imax: width, jmin: 0, jmax: height };
        obj.step(&area, maxit, subs, progress);
        obj
    }
}



#[derive(Clone)]
struct Plotter (ComplexImage<f64>);

impl Plotter {
    <<buddha-plotter-methods>>
}
```

We have a constructor setting the field to all zeros, and the `plot` method using what I know as Cloud-In-Cell method. This basically boils down to linear interpolation. Given $z$ we compute the grid coordinates by scaling and translating. Let $\tilde{z} = x + iy$ be the grid coordinate for our point. First we round down the grid coordinates to the nearest integer value $i = \lfloor x \rfloor$, then we find the  fractional values $x_f = i - \lfloor x \rfloor$ and $y_f = j - \lfloor y \rfloor$, and deposit signal by the following scheme:

|       |         $i$          |    $i + 1$     |
| :---: | :------------------: | :------------: |
|  $j$  | $(1 - x_f)(1 - y_f)$ | $x_f(1 - y_f)$ |
| $j+1$ |    $(1 - x_f)y_f$    |   $x_f y_f$    |

``` {.rust #buddha-plotter-methods}
fn new(width: usize, height: usize, center: Complex<f64>, res: f64) -> Plotter {
    Plotter(ComplexImage {
        pixels: Array2::zeros((width, height)),
        z_min:  center - Complex::new(width as f64 / 2.0, height as f64 / 2.0) * res,
        res
    })
}

fn plot(&mut self, z: Complex<f64>, w: f64) {
    let pz = (z - self.0.z_min) / self.0.res;
    let Complex { re, im } = pz;
    let i = re.floor() as usize;
    let j = im.floor() as usize;
    let (u, v) = self.0.pixels.dim();
    if i > (u-2) || j > (v - 2) {
        return;
    }

    let fx = re - i as f64;
    let fy = im - j as f64;

    self.0.pixels[[i    , j    ]] += w * (1.0 - fx) * (1.0 - fy);
    self.0.pixels[[i + 1, j    ]] += w *        fx  * (1.0 - fy);
    self.0.pixels[[i    , j + 1]] += w * (1.0 - fx) *        fy;
    self.0.pixels[[i + 1, j + 1]] += w *        fx  *        fy;
}
```

## Computing orbits

We can compute orbits using Rust coroutines! These are currently only available in `+nightly` mode. The following generator yields complex values until $|z| > 2$ (here computed as $h\bar{h} > 4$), or a maximum iteration is reached.

``` {.rust #buddha-orbits}
fn mandelbrot_test(c: Complex<f64>, maxit: usize) -> bool {
    let mut z: Complex<f64> = zero();
    for _ in 0..maxit {
        z = z*z + c;
        if (z * z.conj()).re > 4.0 {
            return false;
        }
    }
    return true;
}

struct Orbit {
    points: Vec<Complex<f64>>,
    diverged: bool
}

fn orbit(c: Complex<f64>, maxit: usize) -> Orbit {
    let mut z = Complex::new(0.0, 0.0);
    let mut points = Vec::with_capacity(maxit);
    for _ in 0..maxit {
        z = z*z + c;
        if (z * z.conj()).re > 4.0 {
            return Orbit { points, diverged: true }
        }
        points.push(z);
    }
    return Orbit { points, diverged: false }
}
```

## Subsampling

To accurately compute the Buddhabrot we need to sample a lot of points. I find the best way is to subsample the output grid by a given number.

``` {.rust #buddha-plotter-methods}
fn random_subsample<'a>(&self, mset: &'a ComplexImage<Cell>) -> impl Iterator<Item = Complex<f64>> + 'a {
    let mut rng = rand::thread_rng();
    let top_left = self.0.z_min.clone();
    let res = self.0.res;
    indices(self.0.pixels.dim()).into_iter().filter(
        |(i, j)| mset.pixels[(*i, *j)].face() == SimplexState::Outside
    ).map(
        move |(x, y)| {
            let dx: f64 = rng.gen();
            let dy: f64 = rng.gen();
            Complex::new((x as f64 + dx) * res, (y as f64 + dy) * res) + top_left 
        }
    )
}
```

However, as it turns out, this is also the best place to implement parallelism.

``` {.rust #buddha-plotter-methods}
fn par_subsample(&mut self, mset: &ComplexImage<Cell>, n: usize, maxit: usize) {
    let mut pv = Vec::with_capacity(n*n);
    let w = 1.0 / (n * n) as f64;
    for _ in 0..(n*n) {
        pv.push(self.clone());
    }
    println!("Computing Buddhabrot orbits with {} subsamples per pixel.", n*n);
    pv.par_iter_mut().progress_count((n*n) as u64).for_each(
        |p| {
            for c in p.random_subsample(mset) {
                let o = orbit(c, maxit);
                if o.diverged {
                    o.points.iter().for_each(|&z| p.plot(z, w));
                }
            }
        }
    );
    for sv in pv {
        self.0.pixels += &sv.0.pixels;
    }
}
```

``` {.rust #buddha-plotter-methods}
fn save_pgm(&self, g: GrayscaleMap, fileroot: &String) -> Result<(), Error> {
    let mut file = File::create(format!("{}.pgm", fileroot)).map_err(Error::IO)?;
    let (w, h) = self.0.pixels.dim();
    write!(file, "P5 {} {} 65535\n", w, h).map_err(Error::IO)?;
    let values = match g {
        GrayscaleMap::LINEAR => self.0.pixels.clone(),
        GrayscaleMap::LOG => self.0.pixels.map(|x| (x + 1.0).log10()),
        GrayscaleMap::SQRT => self.0.pixels.map(|x| x.sqrt())
    };
    let max_value = values.iter().reduce(|x, y| if x > y {x} else {y}).ok_or(Error::Value("empty image".to_string()))?;
    values.iter().try_for_each(|x| -> Result<(), Error> {
        let v = (x * 65535.999 / max_value).floor() as u16;
        file.write_all(&v.to_be_bytes()).map_err(Error::IO)?;
        Ok(())
    })?;
    Ok(())
}
```

``` {.rust file=demo/buddhabrot/src/main.rs}
<<buddha-imports>>

#[derive(Debug)]
enum Error {
    IO(std::io::Error),
    Value(String)
}

<<buddha-plotter>>
<<buddha-orbits>>

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum GrayscaleMap {
    LOG, SQRT, LINEAR
}

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[arg(short = 'W', long, default_value_t = 512)]
    width: usize,
    #[arg(short = 'H', long, default_value_t = 512)]
    height: usize,
    #[arg(short, long, default_value_t = 256)]
    maxit: usize,
    #[arg(short, long, default_value_t = 4)]
    subsample: usize,

    #[arg(long)]
    mset: Option<String>,

    /// Grayscale mapping for PGM files
    #[arg(short, long, value_enum, default_value_t = GrayscaleMap::SQRT)]
    grayscale: GrayscaleMap,

    /// Name of Gnuplot output file
    #[arg(long)]
    gnuplot: Option<String>,

    /// Root name of output 16-bit binary PGM files
    #[arg(long)]
    pgm: Option<String>
}

fn main() -> Result<(), Error> 
{
    let args = Cli::parse();
    let mut p = Plotter::new(
        args.width,
        args.height,
        Complex::new(0.0, 0.0),
        4.0 / (args.width as f64));

    let mset = ComplexImage::<Cell>::compute(args.width, args.height, p.0.z_min, p.0.res, args.maxit, args.subsample);
    if let Some(filename) = args.mset {
        mset.write_matrix(&filename, |x| x.get())?;
    }
    p.par_subsample(&mset, args.subsample, args.maxit);
    if let Some(pgm_root) = args.pgm {
        p.save_pgm(args.grayscale, &pgm_root)?;
    }
    if let Some(filename) = args.gnuplot {
        p.0.write_matrix(&filename, |&x| x)?;
    }
    println!("Done!");
    Ok(())
}
```

## Iteration count

The Buddhabrot changes a lot when we vary the maximum number of iterations.

![Buddhabrot for 256, 1024 and 65536 iterations](fig/buddha_iterations.svg)

``` {.make #build}
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
```

``` {.gnuplot #blue-red-palette}
rcol(x) = 0.237 - 2.13*x + 26.92*x**2 - 65.5*x**3 + 63.5*x**4 - 22.36*x**5
gcol(x) = ((0.572 + 1.524*x - 1.811*x**2)/(1 - 0.291*x + 0.1574*x**2))**2
bcol(x) = 1/(1.579 - 4.03*x + 12.92*x**2 - 31.4*x**3 + 48.6*x**4 - 23.36*x**5)
set palette model RGB functions rcol(gray), gcol(gray), bcol(gray)
```

``` {.gnuplot file=demo/plot_buddha_iters.gp}
set term svg size 1000 300
<<blue-red-palette>>
set size ratio -1
set xlabel "Re"
set ylabel "Im"
# set log cb
set xrange [-1.1:-0.1]
set yrange [-0.3:0.7]
unset key; unset colorbox
set multiplot layout 1, 3
set cbrange [1:25]
plot 'data/buddha0010000.dat' matrix nonuniform u 1:2:($3+1) w image
unset ytics; unset ylabel
set cbrange [1:50]
plot 'data/buddha0100000.dat' matrix nonuniform u 1:2:($3+1) w image
set cbrange [1:100]
plot 'data/buddha1000000.dat' matrix nonuniform u 1:2:($3+1) w image
unset multiplot
```

``` {.make file=Makefile}
.RECIPEPREFIX = >

<<build>>
```
