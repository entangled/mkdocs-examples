// ~/~ begin <<docs/buddhabrot.md#demo/buddhabrot/src/main.rs>>[init]
// ~/~ begin <<docs/buddhabrot.md#buddha-imports>>[init]
use rayon::prelude::*;
use rayon::join;
use indicatif::{ProgressBar, ParallelProgressIterator};
use rand::Rng;
use ndarray::{Array2, indices, s};
use num::{Complex, zero};
use clap::{Parser, ValueEnum};
use std::sync::atomic::{AtomicU8,Ordering};

use std::fs::File;
use std::io::Write;
// ~/~ end

#[derive(Debug)]
enum Error {
    IO(std::io::Error),
    Value(String)
}

// ~/~ begin <<docs/buddhabrot.md#buddha-image>>[init]
#[derive(Clone)]
struct ComplexImage<T: Sized> {
    pixels: Array2<T>,
    z_min: Complex<f64>,
    res: f64
}
// ~/~ end
// ~/~ begin <<docs/buddhabrot.md#buddha-image>>[1]
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
// ~/~ end
// ~/~ begin <<docs/buddhabrot.md#buddha-plotter>>[init]
#[derive(Clone)]
struct Plotter (ComplexImage<f64>);

impl Plotter {
    // ~/~ begin <<docs/buddhabrot.md#buddha-plotter-methods>>[init]
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
    // ~/~ end
    // ~/~ begin <<docs/buddhabrot.md#buddha-plotter-methods>>[1]
    fn random_subsample<'a>(&self, mset: &'a ComplexImage<Cell>)
        -> impl Iterator<Item = Complex<f64>> + 'a
    {
        let mut rng = rand::thread_rng();
        let top_left = self.0.z_min.clone();
        let res = self.0.res;
        indices(self.0.pixels.dim()).into_iter().filter(
            |(i, j)| mset.pixels[(*i, *j)].face() != SimplexState::Inside
        ).map(
            move |(x, y)| {
                let dx: f64 = rng.gen();
                let dy: f64 = rng.gen();
                Complex::new((x as f64 + dx) * res, (y as f64 + dy) * res) + top_left 
            }
        )
    }
    // ~/~ end
    // ~/~ begin <<docs/buddhabrot.md#buddha-plotter-methods>>[2]
    fn par_compute(&mut self, mset: &ComplexImage<Cell>, n: usize, maxit: usize) {
        let mut pv = Vec::with_capacity(n*n);
        let w = 1.0 / (n * n) as f64;
        for _ in 0..(n*n) {
            pv.push(self.clone());
        }
        println!("Computing Buddhabrot orbits with {} subsamples per pixel.", n*n);
        pv.par_iter_mut().progress_count((n*n) as u64).for_each(|p| {
            for c in p.random_subsample(mset) {
                let o = orbit(c, maxit);
                if o.diverged {
                    o.points.iter().for_each(|&z| p.plot(z, w));
                }
            }
        });
        for sv in pv {
            self.0.pixels += &sv.0.pixels;
        }
    }
    // ~/~ end
    // ~/~ begin <<docs/buddhabrot.md#buddha-plotter-methods>>[3]
    fn save_pgm(&self, g: GrayscaleMap, filename: &String) -> Result<(), Error> {
        let mut file = File::create(filename).map_err(Error::IO)?;
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
    // ~/~ end
}
// ~/~ end
// ~/~ begin <<docs/buddhabrot.md#buddha-orbits>>[init]
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
// ~/~ end
// ~/~ begin <<docs/buddhabrot.md#buddha-precompute>>[init]
// ~/~ begin <<docs/buddhabrot.md#buddha-mandelbrot>>[init]
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
// ~/~ end
// ~/~ begin <<docs/buddhabrot.md#buddha-simplex-state>>[init]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimplexState {
    Unknown, Outside, Mixed, Inside
}

use SimplexState::*;

impl SimplexState {
    fn from_u8(x: u8) -> Self {
        match x {
            0b00 => Self::Unknown,
            0b01 => Self::Outside,
            0b10 => Self::Mixed,
            0b11 => Self::Inside,
            _    => panic!("invalid number for simplex state")
        }
    }
}
// ~/~ end
// ~/~ begin <<docs/buddhabrot.md#buddha-cell>>[init]
struct Cell (AtomicU8);

impl std::default::Default for Cell {
    fn default() -> Self {
        Cell(AtomicU8::new(0u8))
    }
}

#[allow(dead_code)]
impl Cell {
    // getters
    fn get(&self) -> u8 { self.0.load(Ordering::Relaxed) }
    fn vertex(&self) -> SimplexState { SimplexState::from_u8(self.get()        & 0b00000011 ) }
    fn x_edge(&self) -> SimplexState { SimplexState::from_u8((self.get() >> 2) & 0b00000011 ) }
    fn y_edge(&self) -> SimplexState { SimplexState::from_u8((self.get() >> 4) & 0b00000011 ) }
    fn   face(&self) -> SimplexState { SimplexState::from_u8((self.get() >> 6) & 0b00000011 ) }

    // setters
    fn set_vertex(&self, state: SimplexState) { self.0.fetch_or(state as u8, Ordering::Relaxed); }
    fn set_x_edge(&self, state: SimplexState) { self.0.fetch_or((state as u8) << 2, Ordering::Relaxed); }
    fn set_y_edge(&self, state: SimplexState) { self.0.fetch_or((state as u8) << 4, Ordering::Relaxed); }
    fn   set_face(&self, state: SimplexState) { self.0.fetch_or((state as u8) << 6, Ordering::Relaxed); }
}
// ~/~ end
// ~/~ begin <<docs/buddhabrot.md#buddha-sample-area>>[init]
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
// ~/~ end
// ~/~ end
// ~/~ begin <<docs/buddhabrot.md#buddha-precompute>>[1]
impl ComplexImage<Cell> {
    // ~/~ begin <<docs/buddhabrot.md#buddha-precompute-init>>[init]
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
    // ~/~ end
    // ~/~ begin <<docs/buddhabrot.md#buddha-precompute-step>>[init]
    fn step(&self, area: &SampleArea, maxit: usize, n: usize, progress: ProgressBar) {
        match self.check_boundary(area) {
            SimplexState::Inside => {
                self.mark_area(area, Inside);
                progress.inc(area.size().try_into().unwrap());
                return;
            },
            SimplexState::Outside => {
                self.mark_area(area, Outside);
                progress.inc(area.size().try_into().unwrap());
                return;
            },
            _ => {}
        }

        if area.size() == 1 {
            self.mark_area(area, Mixed);
            progress.inc(1);
            return
        }

        if area.size() == 0 {  // shouldn't happen
            return
        }

        let (a, b) = self.split(area, maxit, n);
        join(|| self.step(&a, maxit, n, progress.clone()), 
             || self.step(&b, maxit, n, progress.clone()));
    }

    fn mark_area(&self, area: &SampleArea, state: SimplexState) {
        for cell in self.pixels.slice(s![area.imin..area.imax,area.jmin..area.jmax]) {
            cell.set_face(state);
        }
    }
    // ~/~ end
    // ~/~ begin <<docs/buddhabrot.md#buddha-precompute-check-boundary>>[init]
    fn check_boundary(&self, area: &SampleArea) -> SimplexState {
        let top = self.pixels.slice(s![area.imin..area.imax,area.jmin]);
        let right = self.pixels.slice(s![area.imax,area.jmin..area.jmax]);
        let bottom = self.pixels.slice(s![area.imin..area.imax,area.jmax]);
        let left = self.pixels.slice(s![area.imin,area.jmin..area.jmax]);

        if  top.iter().all(   |c| c.x_edge() == Inside) &&
            right.iter().all( |c| c.y_edge() == Inside) &&
            bottom.iter().all(|c| c.x_edge() == Inside) &&
            left.iter().all(  |c| c.y_edge() == Inside) {
            return Inside;
        }

        if  top.iter().all(   |c| c.x_edge() == Outside) &&
            right.iter().all( |c| c.y_edge() == Outside) &&
            bottom.iter().all(|c| c.x_edge() == Outside) &&
            left.iter().all(  |c| c.y_edge() == Outside) {
            return Outside; 
        }

        Mixed
    }
    // ~/~ end
    // ~/~ begin <<docs/buddhabrot.md#buddha-precompute-split>>[init]
    fn horizontal_split(&self, area: &SampleArea, maxit: usize, n: usize) -> (SampleArea, SampleArea) {
        let b = (area.jmin + area.jmax) / 2;
        let y = self.z_min.im + (b as f64) * self.res;
        for a in area.imin..area.imax {
            let x = self.z_min.re + (a as f64) * self.res;
            let c = (0..n).map(
                |k| Complex::new(x + (k as f64 / n as f64) * self.res, y));
            let res: Vec<bool> = c.map(|c| mandelbrot_test(c, maxit)).collect();
            if res.iter().all(|&x| x) {
                self.pixels[(a, b)].set_x_edge(Inside);
            } else if res.iter().any(|&x| x) {
                self.pixels[(a, b)].set_x_edge(Mixed);
            } else {
                self.pixels[(a, b)].set_x_edge(Outside);
            }
        }
        ( SampleArea { jmax: b, ..*area }, SampleArea { jmin: b, ..*area } )
    }

    fn vertical_split(&self, area: &SampleArea, maxit: usize, n: usize) -> (SampleArea, SampleArea) {
        let a = (area.imin + area.imax) / 2;
        let x = self.z_min.re + (a as f64) * self.res;
        for b in area.jmin..area.jmax {
            let y = self.z_min.im + (b as f64) * self.res;
            let c = (0..n).map(
                |k| Complex::new(x, y + (k as f64 / n as f64) * self.res));
            let res: Vec<bool> = c.map(|c| mandelbrot_test(c, maxit)).collect();
            if res.iter().all(|&x| x) {
                self.pixels[(a, b)].set_y_edge(Inside)
            } else if res.iter().any(|&x| x) {
                self.pixels[(a, b)].set_y_edge(Mixed);
            } else {
                self.pixels[(a, b)].set_y_edge(Outside);
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
    // ~/~ end
}
// ~/~ end

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
    p.par_compute(&mset, args.subsample, args.maxit);
    if let Some(pgm_root) = args.pgm {
        p.save_pgm(args.grayscale, &pgm_root)?;
    }
    if let Some(filename) = args.gnuplot {
        p.0.write_matrix(&filename, |&x| x)?;
    }
    println!("Done!");
    Ok(())
}
// ~/~ end