
use std::sync::{Arc, Mutex};

use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

struct Consts {
    time0: usize,
    swing0: usize,
    note0: usize,
    arc0: usize,
    bomb0: usize,
    wall0: usize,
    dur0: usize,
    events_end: usize,
    size: usize,
    eos: usize,
    ticks: i64,
    kinds: [i64; 5],
    wall_cells: Vec<(u16, u16)>,
    wall_head: Vec<u8>,
    durations: Vec<i64>,
    dir_angle: [i64; 8],
    dir_sin: [f64; 8],
    dir_cos: [f64; 8],
    slide_ok: Vec<bool>,
    clash: Vec<bool>,
    slide_turn: i64,
    forehand: [u16; 2],
    backhand: [u16; 2],
    opening: u16,
    down: u16,
    up: u16,
    hand_cols: [u8; 2],
    max_stack: i64,
    arc_cap: f64,
    arc_span: i64,
    loop_periods: usize,
    loop_match: f64,
    loop_penalty: f64,
    variety_beats: i64,
    variety_notes: usize,
    variety_floor: usize,
    variety_penalty: f64,
    dry_base: f64,
    dry_allowed: usize,
    dry_window: usize,
    dry_grow: usize,
    double_share: f64,
    double_from: i64,
    double_push: f64,
    double_max: f64,
    air_rate: f64,
    air_from: i64,
    air_push: f64,
    air_max: f64,
    hold_from: f64,
    hold_grow: f64,
    hold_max: f64,
    obj_walls: i64,
    obj_bombs: i64,
    obj_arcs: i64,
    no_note: i64,
}

fn get<'py, T: for<'a> FromPyObject<'a, 'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    match d.get_item(key)? {
        Some(v) => v.extract::<T>().map_err(|_| PyValueError::new_err(format!("constant {key}: wrong type"))),
        None => Err(PyValueError::new_err(format!("missing constant {key}"))),
    }
}

fn bits(v: &[i64]) -> u16 {
    v.iter().fold(0, |m, &d| m | 1 << d)
}

#[pyclass(frozen, module = "bsgen._rules")]
struct NativeConsts {
    c: Arc<Consts>,
}

#[pymethods]
impl NativeConsts {
    #[new]
    fn new(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let cells: Vec<(Vec<i64>, Vec<i64>)> = get(d, "wall_cells")?;
        let v8 = |key: &str| -> PyResult<[f64; 8]> {
            let v: Vec<f64> = get(d, key)?;
            v.try_into().map_err(|_| PyValueError::new_err(format!("{key}: 8 values")))
        };
        let angle: Vec<i64> = get(d, "dir_angle")?;
        let kinds: Vec<i64> = get(d, "kinds")?;
        let cols: Vec<Vec<i64>> = get(d, "hand_cols")?;
        let c = Consts {
            time0: get(d, "time0")?,
            swing0: get(d, "swing0")?,
            note0: get(d, "note0")?,
            arc0: get(d, "arc0")?,
            bomb0: get(d, "bomb0")?,
            wall0: get(d, "wall0")?,
            dur0: get(d, "dur0")?,
            events_end: get(d, "events_end")?,
            size: get(d, "size")?,
            eos: get(d, "eos")?,
            ticks: get(d, "ticks")?,
            kinds: kinds.try_into().map_err(|_| PyValueError::new_err("kinds: 5 values"))?,
            wall_cells: cells.iter().map(|(a, b)| (bits(a), bits(b))).collect(),
            wall_head: get(d, "wall_head")?,
            durations: get(d, "durations")?,
            dir_angle: angle.try_into().map_err(|_| PyValueError::new_err("dir_angle: 8 values"))?,
            dir_sin: v8("dir_sin")?,
            dir_cos: v8("dir_cos")?,
            slide_ok: get(d, "slide_ok")?,
            clash: get(d, "clash")?,
            slide_turn: get(d, "slide_turn")?,
            forehand: get::<Vec<Vec<i64>>>(d, "forehand")?.iter().map(|v| bits(v)).collect::<Vec<_>>().try_into().map_err(|_| PyValueError::new_err("forehand: 2 hands"))?,
            backhand: get::<Vec<Vec<i64>>>(d, "backhand")?.iter().map(|v| bits(v)).collect::<Vec<_>>().try_into().map_err(|_| PyValueError::new_err("backhand: 2 hands"))?,
            opening: bits(&get::<Vec<i64>>(d, "opening")?),
            down: bits(&get::<Vec<i64>>(d, "down")?),
            up: bits(&get::<Vec<i64>>(d, "up")?),
            hand_cols: [bits(&cols[0]) as u8, bits(&cols[1]) as u8],
            max_stack: get(d, "max_stack")?,
            arc_cap: get(d, "arc_cap")?,
            arc_span: get(d, "arc_span")?,
            loop_periods: get(d, "loop_periods")?,
            loop_match: get(d, "loop_match")?,
            loop_penalty: get(d, "loop_penalty")?,
            variety_beats: get(d, "variety_beats")?,
            variety_notes: get(d, "variety_notes")?,
            variety_floor: get(d, "variety_floor")?,
            variety_penalty: get(d, "variety_penalty")?,
            dry_base: get(d, "dry_base")?,
            dry_allowed: get(d, "dry_allowed")?,
            dry_window: get(d, "dry_window")?,
            dry_grow: get(d, "dry_grow")?,
            double_share: get(d, "double_share")?,
            double_from: get(d, "double_from")?,
            double_push: get(d, "double_push")?,
            double_max: get(d, "double_max")?,
            air_rate: get(d, "air_rate")?,
            air_from: get(d, "air_from")?,
            air_push: get(d, "air_push")?,
            air_max: get(d, "air_max")?,
            hold_from: get(d, "hold_from")?,
            hold_grow: get(d, "hold_grow")?,
            hold_max: get(d, "hold_max")?,
            obj_walls: get(d, "obj_walls")?,
            obj_bombs: get(d, "obj_bombs")?,
            obj_arcs: get(d, "obj_arcs")?,
            no_note: get(d, "no_note")?,
        };
        Ok(NativeConsts { c: Arc::new(c) })
    }
}

fn turn(a: i64, b: i64) -> i64 {
    ((a - b + 180).rem_euclid(360) - 180).abs()
}

fn left(v: &[i64], x: i64) -> usize {
    v.partition_point(|&t| t < x)
}

fn right(v: &[i64], x: i64) -> usize {
    v.partition_point(|&t| t <= x)
}

type Sym = (i64, i64, i64, i64);

#[derive(Clone)]
struct Loops {
    limit: i64,
    dry: f64,
    copy: Option<(i64, i64)>,
    sym: Vec<Sym>,
    ticks: Vec<i64>,
    codes: Vec<i64>,
    expect: Vec<(i64, i64, i64, i64, f64)>,
    overused: Vec<(i64, i64, i64, f64)>,
    repeats: Vec<(i64, i64, i64, i64, f64)>,
    stale: bool,
}

impl Loops {
    fn add(&mut self, c: &Consts, tick: i64, hand: i64, cell: i64, d: i64, check: bool) {
        let gap = self.ticks.last().map_or(0, |&t| tick - t);
        self.sym.push((hand, cell, d, gap));
        self.ticks.push(tick);
        self.codes.push(((hand * 12 + cell) * 9 + d) * 1_000_000 + gap);
        if self.sym.len() > (4 * c.loop_periods).max(c.dry_window) + 64 {
            self.sym.drain(..64);
            self.ticks.drain(..64);
            self.codes.drain(..64);
        }
        self.stale = self.dry > 0.0;
        self.expect.clear();
        self.overused.clear();
        if check {
            self.check(c);
        }
    }

    fn check(&mut self, c: &Consts) {
        self.expect.clear();
        self.overused.clear();
        if self.limit <= 0 || self.ticks.is_empty() {
            return;
        }
        let tick = *self.ticks.last().unwrap();
        let n = self.ticks.len();
        let since = right(&self.ticks, tick - c.variety_beats * c.ticks);
        let mut recent: Vec<(i64, i64, i64)> = Vec::new();
        for s in &self.sym[since..] {
            let k = (s.0, s.1, s.2);
            if !recent.contains(&k) {
                recent.push(k);
            }
        }
        if n - since >= c.variety_notes && recent.len() < c.variety_floor {
            let amount = c.variety_penalty * (c.variety_floor - recent.len()) as f64;
            self.overused = recent.iter().map(|&(h, cl, d)| (h, cl, d, amount)).collect();
        }
        let first = left(&self.ticks, tick - self.limit);
        if n - first < 4 {
            return;
        }
        for p in 1..=c.loop_periods {
            if n - first < 2 * p || first < p {
                continue;
            }
            let same = (first..n).filter(|&i| self.codes[i] == self.codes[i - p]).count();
            if (same as f64) / ((n - first) as f64) < c.loop_match {
                continue;
            }
            let mut start = first;
            while start > p && self.sym[start - 1] == self.sym[start - 1 - p] {
                start -= 1;
            }
            let over = (tick - self.ticks[start - p] - self.limit) as f64 / c.ticks as f64;
            let (h, cl, dd, g) = self.sym[n - p];
            self.expect.push((tick + g, h, cl, dd, c.loop_penalty + over.max(0.0)));
        }
    }

    fn find_repeats(&self, c: &Consts) -> Vec<(i64, i64, i64, i64, f64)> {
        let m = self.codes.len();
        let n = m.min(c.dry_window);
        let s = &self.codes[m - n..];
        if n <= c.dry_allowed {
            return Vec::new();
        }
        let mut alive: Vec<usize> = (0..n - 1).collect();
        let mut length = vec![0usize; n - 1];
        for k in 0..c.dry_allowed + c.dry_grow {
            if k > n - 1 {
                alive.clear();
            } else {
                let now = s[n - 1 - k];
                alive.retain(|&a| a >= k && s[a - k] == now);
            }
            if alive.is_empty() {
                break;
            }
            for &a in &alive {
                length[a] += 1;
            }
        }
        let mut best: Vec<((i64, i64, i64, i64), f64)> = Vec::new();
        let last = *self.ticks.last().unwrap();
        for j in 0..n - 1 {
            if length[j] < c.dry_allowed {
                continue;
            }
            if let Some((c0, c1)) = self.copy {
                let t = self.ticks[self.ticks.len() - n + j];
                if c0 <= t && t < c1 {
                    continue;
                }
            }
            let (h, cl, d, g) = self.sym[self.sym.len() - n + j + 1];
            let key = (last + g, h, cl, d);
            let v = self.dry * c.dry_base.powf((length[j] - c.dry_allowed) as f64);
            match best.iter_mut().find(|(k, _)| *k == key) {
                Some(e) => e.1 = e.1.max(v),
                None => best.push((key, 0.0f64.max(v))),
            }
        }
        best.into_iter().map(|((t, h, cl, d), a)| (t, h, cl, d, a)).collect()
    }
}

#[derive(Clone)]
struct State {
    c: Arc<Consts>,
    end: i64,
    swings: bool,
    objects: i64,
    turn: f64,
    hold: f64,
    times: Option<Vec<i64>>,
    behind: i64,
    double_gap: i64,
    double: Option<i64>,
    swing_gap: i64,
    slide_gap: i64,
    uncross: bool,
    cell_gap: i64,
    cell_tick: [Option<i64>; 12],
    bomb_tick: [Option<i64>; 12],
    near_gap: i64,
    wall_gap: i64,
    vision_gap: i64,
    front_tick: [Option<(i64, i64)>; 2],
    cols: [Option<(i64, u8)>; 2],
    moments_n: i64,
    doubles_n: i64,
    swings_n: i64,
    air_free_n: i64,
    head: [Option<i64>; 2],
    rest: [Option<i64>; 2],
    owe: [bool; 2],
    loops: Loops,
    tick: Option<i64>,
    t0: i64,
    face: [Option<bool>; 2],
    need_note: Option<usize>,
    need_dur: bool,
    swing_tick: [Option<i64>; 2],
    air: [bool; 2],
    cells: Vec<(u8, u8)>,
    after_time: bool,
    opened: [bool; 2],
    first_dir: [Option<i64>; 2],
    stack: [i64; 2],
    prev_moment: [Option<i64>; 2],
    notes_now: [Vec<i64>; 2],
    prev_note: [i64; 2],
    arc_done: [bool; 2],
    note_ticks: Vec<i64>,
    arc_ticks: Vec<i64>,
    objects_now: bool,
    walls: Vec<(i64, i64, u16, u16, u8)>,
    pending_wall: usize,
    need_time: bool,
    swing_dir: [Option<(i64, i64)>; 2],
    last_swing: [Option<(i64, i64)>; 2],
    hit: Option<i64>,
    run_gap: Option<i64>,
    run_from: Option<i64>,
}

const ALL_CELLS: u16 = (1 << 12) - 1;
const ALL_DIRS: u16 = (1 << 9) - 1;

fn ones(m: u16, n: usize) -> impl Iterator<Item = usize> {
    (0..n).filter(move |&i| m >> i & 1 == 1)
}

impl State {
    fn note(&self, h: usize, cell: usize, d: usize) -> usize {
        self.c.note0 + (h * 12 + cell) * 9 + d
    }

    fn swing(&self, h: usize, forehand: bool, air: bool) -> usize {
        self.c.swing0 + h * 4 + forehand as usize * 2 + air as usize
    }

    fn in_times(&self, t: Option<i64>) -> bool {
        match (&self.times, t) {
            (Some(v), Some(t)) => v.binary_search(&t).is_ok(),
            _ => false,
        }
    }

    fn next_swing(&mut self, h: usize, air: bool) {
        self.last_swing[h] = if air { None } else { self.swing_dir[h] };
        self.swing_dir[h] = None;
    }

    fn set_rest(&mut self, h: usize, d: i64) {
        let c = &self.c;
        self.rest[h] = if c.down >> d & 1 == 1 || d == 8 && self.face[h] == Some(true) {
            Some(0)
        } else if c.up >> d & 1 == 1 || d == 8 && self.face[h] == Some(false) {
            Some(2)
        } else {
            None
        };
    }

    fn swung(&mut self, h: usize, air: bool) {
        self.swings_n += 1;
        self.air_free_n += (air && !self.owe[h] && (self.times.is_none() || self.in_times(self.tick))) as i64;
        self.rest[h] = if air { self.rest[h].map(|r| 2 - r) } else { None };
        self.owe[h] = false;
    }

    fn bomb(&mut self, cell: i64) {
        for h in 0..2 {
            if let Some(r) = self.rest[h]
                && cell / 4 == r
                && self.c.hand_cols[h] >> (cell % 4) & 1 == 1
                && !(self.swing_tick[h] == self.tick && !self.air[h])
            {
                self.owe[h] = true;
            }
        }
    }

    fn late(&self, h: usize) -> bool {
        self.owe[h] && self.prev_moment[h].is_some_and(|p| self.tick.unwrap() - p > self.c.ticks)
    }

    fn ahead(&mut self, h: usize, cell: i64) {
        let (Some(head), Some((d, _))) = (self.head[h], self.swing_dir[h]) else {
            self.head[h] = Some(cell);
            return;
        };
        if self.stack[h] == 0 {
            self.head[h] = Some(cell);
            return;
        }
        let (s, co) = (self.c.dir_sin[d as usize], self.c.dir_cos[d as usize]);
        let along = |c: i64| (c % 4) as f64 * s - (c / 4) as f64 * co;
        if along(cell) > along(head) {
            self.head[h] = Some(cell);
        }
    }

    fn slide(&self, h: usize, cells: u16, dirs: u16) -> (u16, u16) {
        let (Some(head), Some((a, _))) = (self.head[h], self.swing_dir[h]) else {
            return (cells, dirs);
        };
        if self.stack[h] != 0 {
            return (cells, dirs);
        }
        let c = &self.c;
        let (head, a) = (head as usize, a as usize);
        let mut keep = 0u16;
        for cl in ones(cells, 12) {
            if cl != head && c.slide_ok[(head * 12 + cl) * 8 + a] {
                keep |= 1 << cl;
            }
        }
        let mut kd = 0u16;
        for d in ones(dirs, 9) {
            if d == 8 || turn(c.dir_angle[d], c.dir_angle[a]) <= c.slide_turn {
                kd |= 1 << d;
            }
        }
        (keep, kd)
    }

    fn hit_now(&mut self) {
        let tick = self.tick.unwrap();
        if self.hit == Some(tick) {
            return;
        }
        if let Some(hit) = self.hit
            && Some(tick - hit) != self.run_gap
        {
            self.run_gap = Some(tick - hit);
            self.run_from = Some(hit);
        }
        self.hit = Some(tick);
    }

    fn new_moment(&mut self, tick: i64) {
        if self.tick.is_some() {
            if self.stack[0] != 0 || self.stack[1] != 0 {
                self.moments_n += 1;
                self.doubles_n += (self.stack[0] != 0 && self.stack[1] != 0) as i64;
            }
            for h in 0..2 {
                if self.stack[h] != 0 {
                    self.prev_moment[h] = self.tick;
                }
                if let Some(&m) = self.notes_now[h].iter().max() {
                    self.prev_note[h] = m;
                }
            }
        }
        self.notes_now = [Vec::new(), Vec::new()];
        self.tick = Some(tick);
        self.cells.clear();
        self.objects_now = false;
        self.stack = [0, 0];
        self.first_dir = [None, None];
        self.arc_done = [false, false];
        let keep = self.behind.max(self.wall_gap);
        self.walls.retain(|w| w.1 + keep > tick);
    }

    fn add_wall(&mut self, shape: usize, dur: i64) {
        let t = self.tick.unwrap();
        let (cells, lanes) = self.c.wall_cells[shape];
        self.walls.push((t, t + dur, cells, lanes, self.c.wall_head[shape]));
    }

    fn add_cell(&mut self, h: u8, cell: u8) {
        if !self.cells.contains(&(h, cell)) {
            self.cells.push((h, cell));
        }
    }

    fn col(&mut self, h: usize, cell: i64) {
        let t = self.tick.unwrap();
        if cell == 5 || cell == 6 {
            self.front_tick[h] = Some((t, cell));
        }
        match &mut self.cols[h] {
            Some((ct, m)) if *ct == t => *m |= 1 << (cell % 4),
            slot => *slot = Some((t, 1 << (cell % 4))),
        }
    }

    fn covered(&self, behind: i64, notes: bool) -> u16 {
        let t = self.tick.unwrap();
        let mut out = 0;
        for &(start, end, cells, lanes, _) in &self.walls {
            if start <= t && t < end {
                out |= if notes { lanes } else { cells };
            } else if end <= t && t < end + behind {
                out |= lanes;
            }
        }
        out
    }

    fn free(&self, behind: i64, notes: bool) -> u16 {
        let t = self.tick.unwrap();
        let mut out = ALL_CELLS & !self.cells.iter().fold(0u16, |m, &(_, c)| m | 1 << c) & !self.covered(behind, notes);
        if self.cell_gap != 0 {
            let near = |v: &[Option<i64>; 12]| {
                (0..12).filter(|&c| v[c].is_some_and(|x| 0 < t - x && t - x < self.cell_gap)).fold(0u16, |m, c| m | 1 << c)
            };
            out &= !near(&self.cell_tick);
            if notes {
                out &= !near(&self.bomb_tick);
            }
        }
        out
    }

    fn sides(&self, h: usize, cells: u16, first: bool) -> u16 {
        let t = self.tick.unwrap();
        let other: Vec<i64> = self.cells.iter().filter(|&&(g, _)| g as usize == 1 - h).map(|&(_, c)| c as i64 % 4).collect();
        let near = match self.cols[1 - h] {
            Some((ct, m)) if self.near_gap != 0 && 0 < t - ct && t - ct < self.near_gap => Some(m),
            _ => None,
        };
        let front = self.front_tick[1 - h].filter(|&(f, _)| self.vision_gap != 0 && 0 < t - f && t - f < self.vision_gap);
        if other.is_empty() && near.is_none() && front.is_none() {
            return cells;
        }
        let own: Vec<i64> = self.cells.iter().filter(|&&(g, _)| g as usize == h).map(|&(_, c)| c as i64 % 4).collect();
        let fine = |c: i64| -> bool {
            let col = c % 4;
            if front.is_some_and(|(_, cell)| (col <= 1) == (cell == 5)) {
                return false;
            }
            if let Some(m) = near {
                let lo = m.trailing_zeros() as i64;
                let hi = 7 - m.leading_zeros() as i64;
                if if h == 0 { col >= lo } else { col <= hi } {
                    return false;
                }
            }
            if other.is_empty() {
                return true;
            }
            let lo = *other.iter().min().unwrap();
            let hi = *other.iter().max().unwrap();
            let nmin = own.iter().copied().chain([col]).min().unwrap();
            let nmax = own.iter().copied().chain([col]).max().unwrap();
            if nmin < lo && lo < nmax || lo < nmin && nmin < hi {
                return false;
            }
            !self.uncross || if h == 0 { col <= lo } else { col >= hi }
        };
        let kept = ones(cells, 12).filter(|&c| fine(c as i64)).fold(0u16, |m, c| m | 1 << c);
        if kept == 0 && first { cells } else { kept }
    }

    fn line(&self, h: usize, cells: u16) -> u16 {
        let own: Vec<i64> = self.notes_now[h].iter().map(|n| n / 9).collect();
        if own.len() < 2 {
            return cells;
        }
        let (x0, y0, x1, y1) = (own[0] % 4, own[0] / 4, own[1] % 4, own[1] / 4);
        ones(cells, 12)
            .filter(|&c| {
                let c = c as i64;
                (c % 4 - x0) * (y1 - y0) == (c / 4 - y0) * (x1 - x0)
            })
            .fold(0u16, |m, c| m | 1 << c)
    }

    fn put(&self, ok: &mut [bool], h: usize, cells: u16, dirs: u16, first: bool) {
        let c = &self.c;
        let clear = |cl: usize, d: usize| {
            self.notes_now[1 - h].iter().all(|&o| {
                let (o, od) = ((o / 9) as usize, (o % 9) as usize);
                !c.clash[(cl * 12 + o) * 9 + d] && !c.clash[(o * 12 + cl) * 9 + od]
            })
        };
        let any = ones(cells, 12).any(|cl| ones(dirs, 9).any(|d| clear(cl, d)));
        for cl in ones(cells, 12) {
            for d in ones(dirs, 9) {
                if (first && !any) || clear(cl, d) {
                    ok[self.note(h, cl, d)] = true;
                }
            }
        }
    }

    fn fit(&self, h: usize, dirs: u16) -> u16 {
        if self.stack[h] >= self.c.max_stack {
            return 0;
        }
        match self.first_dir[h] {
            None | Some(8) => dirs,
            Some(d0) => ones(dirs, 9)
                .filter(|&d| d == 8 || turn(self.c.dir_angle[d], self.c.dir_angle[d0 as usize]) <= 45)
                .fold(0u16, |m, d| m | 1 << d),
        }
    }

    fn between(&self) -> bool {
        self.times.is_some() && self.tick.is_some() && !self.need_time && !self.in_times(self.tick)
    }

    fn way_on(&self) -> usize {
        let Some(times) = &self.times else { return self.c.eos };
        let lo = self.tick.map_or(0, |t| t + 1 - self.t0);
        let j = left(times, self.t0 + lo.max(0));
        if j < times.len() && times[j] < self.end {
            (self.c.time0 as i64 + times[j] - self.t0) as usize
        } else {
            self.c.eos
        }
    }

    fn mask(&self) -> Vec<bool> {
        let ok = self.mask_with(false);
        if self.times.is_some() && !ok.iter().any(|&x| x) {
            return self.mask_with(true);
        }
        ok
    }

    fn mask_with(&self, waive: bool) -> Vec<bool> {
        let c = &self.c;
        let mut ok = vec![false; c.size];
        let fill = |ok: &mut Vec<bool>, a: i64, b: i64| {
            let (a, b) = (a.clamp(0, c.size as i64) as usize, b.clamp(0, c.size as i64) as usize);
            if a < b {
                ok[a..b].fill(true);
            }
        };
        if self.need_dur {
            ok[c.dur0..c.events_end].fill(true);
            return ok;
        }
        if let Some(h) = self.need_note {
            let mut dirs = if self.face[h] == Some(true) { c.forehand[h] } else { c.backhand[h] };
            if !self.opened[h] {
                dirs &= c.opening;
            }
            self.put(&mut ok, h, self.sides(h, self.free(self.behind, true), true), self.fit(h, dirs), true);
            return ok;
        }
        let lo = self.tick.map_or(0, |t| t + 1 - self.t0);
        let hi = self.end - self.t0;
        let time0 = c.time0 as i64;
        let owed = self.times.is_some() && !waive && !self.need_time && self.in_times(self.tick)
            && !(self.stack[0] != 0 || self.stack[1] != 0);
        let between = self.between();
        let filled = self.objects_now || self.tick == self.swing_tick[0] || self.tick == self.swing_tick[1];
        if !self.after_time && !owed && !(between && !filled && !waive) {
            if self.times.is_none() {
                fill(&mut ok, time0 + lo.max(0), time0 + hi);
                ok[c.eos] = true;
            } else {
                let way = self.way_on();
                if self.objects & (c.obj_bombs | c.obj_walls) != 0 && !waive {
                    let stop = if way != c.eos { way as i64 - time0 } else { hi };
                    fill(&mut ok, time0 + lo.max(0), time0 + stop);
                }
                ok[way] = true;
            }
        }
        if self.tick.is_none() || self.need_time {
            return ok;
        }
        let tick = self.tick.unwrap();
        let free = self.free(0, false);
        let note_free = self.free(self.behind, true);
        if owed {
        } else if self.objects & c.obj_bombs != 0 && (self.times.is_none() || between) {
            for cell in ones(free, 12) {
                ok[c.bomb0 + cell] = true;
            }
        }
        if self.objects & c.obj_walls != 0 && !owed {
            let bombs = self.cells.iter().filter(|x| x.0 == 2).fold(0u16, |m, x| m | 1 << x.1);
            let notes = self.cells.iter().filter(|x| x.0 != 2).fold(0u16, |m, x| m | 1 << x.1);
            let closed = self.tick.map_or(0, |t| {
                self.walls.iter().filter(|w| w.0 <= t && t < w.1 + self.wall_gap).fold(0u8, |m, w| m | w.4)
            });
            for (sh, &(cells, lanes)) in c.wall_cells.iter().enumerate() {
                let head = c.wall_head[sh];
                ok[c.wall0 + sh] = !(cells & bombs != 0
                    || lanes & notes != 0
                    || head == 3
                    || (head & !closed & 3 != 0 && closed != 0));
            }
        }
        if between && !self.objects_now {
            for h in 0..2 {
                if self.swing_tick[h] != self.tick
                    && let Some(f) = self.face[h]
                {
                    ok[self.swing(h, !f, true)] = true;
                }
            }
        }
        if self.objects_now || between {
            return ok;
        }
        if self.objects & c.obj_arcs != 0 {
            let since = tick - c.arc_span * c.ticks;
            let arcs = self.arc_ticks.len() - left(&self.arc_ticks, since) + 1;
            let notes = self.note_ticks.len() - left(&self.note_ticks, since);
            let room = arcs as f64 <= c.arc_cap * notes as f64;
            for h in 0..2 {
                ok[c.arc0 + h] = room && self.stack[h] > 0 && self.prev_moment[h].is_some() && !self.arc_done[h];
            }
        }
        if !self.swings {
            for h in 0..2 {
                if !self.arc_done[h] {
                    let dirs = self.fit(h, if self.opened[h] { ALL_DIRS } else { c.opening });
                    for cell in ones(note_free, 12) {
                        for d in ones(dirs, 9) {
                            ok[self.note(h, cell, d)] = true;
                        }
                    }
                }
            }
            return ok;
        }
        let no_double = self.double.is_some_and(|d| d != tick && tick - d < self.double_gap);
        let soon = |h: usize| self.stack[h] == 0 && self.prev_moment[h].is_some_and(|p| tick - p < self.swing_gap);
        for h in 0..2 {
            if self.arc_done[h] {
                continue;
            }
            if no_double && self.swing_tick[1 - h] == self.tick || soon(h) {
            } else if self.swing_tick[h] != self.tick && note_free != 0 {
                match self.face[h] {
                    None => ok[self.swing(h, true, false)] = true,
                    Some(face) => {
                        let f = !face;
                        ok[self.swing(h, f, true)] = true;
                        ok[self.swing(h, if self.late(h) { !f } else { f }, false)] = true;
                    }
                }
            } else if self.swing_tick[h] != self.tick
                && let Some(f) = self.face[h]
            {
                ok[self.swing(h, !f, true)] = true;
            }
            let late_slide = self.slide_gap != 0
                && self.stack[h] == 0
                && self.prev_moment[h].is_some_and(|p| tick - p > self.slide_gap);
            if !self.air[h]
                && !late_slide
                && let Some(f) = self.face[h]
            {
                let (cells, dirs) = self.slide(
                    h,
                    self.line(h, self.sides(h, note_free, false)),
                    self.fit(h, if f { c.forehand[h] } else { c.backhand[h] }),
                );
                self.put(&mut ok, h, cells, dirs, false);
            }
        }
        ok
    }

    fn loop_penalty(&mut self) -> Vec<(usize, f64)> {
        let c = self.c.clone();
        if self.loops.stale {
            self.loops.repeats = self.loops.find_repeats(&c);
            self.loops.stale = false;
        }
        let mut out: Vec<(usize, f64)> =
            self.loops.overused.iter().map(|&(h, cl, d, a)| (self.note(h as usize, cl as usize, d as usize), a)).collect();
        for &(t, h, cl, d, a) in &self.loops.repeats {
            if self.tick == Some(t) {
                out.push((self.note(h as usize, cl as usize, d as usize), a));
            }
        }
        for &(t, h, cl, d, a) in &self.loops.expect {
            if self.tick == Some(t) {
                out.push((self.note(h as usize, cl as usize, d as usize), a));
            } else if let Some(tick) = self.tick
                && tick < t
                && t < self.end
                && !self.after_time
            {
                out.push(((c.time0 as i64 + t - self.t0) as usize, a));
            }
        }
        out
    }

    fn turn_penalty(&self) -> Vec<(usize, f64)> {
        let Some(h) = self.need_note else { return Vec::new() };
        let Some((d0, t)) = self.last_swing[h] else { return Vec::new() };
        if self.turn == 0.0 {
            return Vec::new();
        }
        let c = &self.c;
        let amount = self.turn / 1.0f64.max((self.tick.unwrap() - t) as f64 / c.ticks as f64);
        let back = (c.dir_angle[d0 as usize] + 180).rem_euclid(360);
        let dirs: Vec<usize> = (0..8).filter(|&d| turn(c.dir_angle[d], back) >= 90).collect();
        let mut out = Vec::new();
        for cell in 0..12 {
            for &d in &dirs {
                out.push((self.note(h, cell, d), amount));
            }
        }
        out
    }

    fn hold_penalty(&self) -> Vec<(usize, f64)> {
        let c = &self.c;
        let Some(gap) = self.run_gap else { return Vec::new() };
        if self.hold == 0.0 || self.need_note.is_some() || self.need_dur || self.after_time || gap > c.ticks {
            return Vec::new();
        }
        let (hit, from) = (self.hit.unwrap(), self.run_from.unwrap());
        let over = (hit - from) as f64 / c.ticks as f64 - c.hold_from;
        let k = hit + gap - self.t0;
        let tick = self.tick.unwrap();
        if over < 0.0 || !(tick + 1 - self.t0 <= k && k < self.end - self.t0) {
            return Vec::new();
        }
        vec![((c.time0 as i64 + k) as usize, self.hold * (1.0 + over / c.hold_grow).min(c.hold_max))]
    }

    fn double_penalty(&self) -> Vec<(usize, f64)> {
        let c = &self.c;
        if self.moments_n < c.double_from || self.need_note.is_some() || self.tick.is_none() {
            return Vec::new();
        }
        let over = self.doubles_n as f64 / self.moments_n as f64 - c.double_share;
        if over <= 0.0 {
            return Vec::new();
        }
        let amount = (c.double_push * over / 0.1).min(c.double_max);
        let mut out = Vec::new();
        for h in 0..2 {
            if self.stack[1 - h] != 0 && self.stack[h] == 0 {
                for f in [false, true] {
                    out.push((self.swing(h, f, false), amount));
                }
            }
        }
        out
    }

    fn air_penalty(&self) -> Vec<(usize, f64)> {
        let c = &self.c;
        if self.swings_n < c.air_from
            || self.need_note.is_some()
            || self.need_time
            || self.objects_now
            || self.tick.is_none()
            || (self.times.is_some() && !self.in_times(self.tick))
        {
            return Vec::new();
        }
        let over = self.air_free_n as f64 / self.swings_n as f64 - c.air_rate;
        if over <= 0.0 {
            return Vec::new();
        }
        let amount = (c.air_push * over / 0.01).min(c.air_max);
        let mut out = Vec::new();
        for h in 0..2 {
            if !self.owe[h] {
                for f in [false, true] {
                    out.push((self.swing(h, f, true), amount));
                }
            }
        }
        out
    }

    fn penalties(&mut self) -> Vec<(usize, f64)> {
        let mut out = self.loop_penalty();
        out.extend(self.turn_penalty());
        out.extend(self.hold_penalty());
        out.extend(self.double_penalty());
        out.extend(self.air_penalty());
        out
    }

    fn reset_merge(&self) -> Vec<(usize, usize)> {
        if self.tick.is_none() || self.need_note.is_some() || self.need_time || self.objects_now {
            return Vec::new();
        }
        let mut out = Vec::new();
        for h in 0..2 {
            if let Some(f) = self.face[h]
                && self.swing_tick[h] != self.tick
                && self.late(h)
            {
                out.push((self.swing(h, f, false), self.swing(h, !f, false)));
            }
        }
        out
    }

    fn push(&mut self, k: usize) {
        let c = self.c.clone();
        if c.time0 <= k && k < c.swing0 {
            self.new_moment(self.t0 + (k - c.time0) as i64);
            self.after_time = true;
            self.need_time = false;
            return;
        }
        self.after_time = false;
        if c.swing0 <= k && k < c.note0 {
            let j = k - c.swing0;
            let h = j / 4;
            let (fh, air) = (j / 2 % 2 == 1, j % 2 == 1);
            let same = self.face[h] == Some(fh) && !air;
            self.face[h] = Some(fh);
            self.air[h] = air;
            self.swing_tick[h] = self.tick;
            if self.swing_tick[1 - h] == self.tick {
                self.double = self.tick;
            }
            self.need_note = if air { None } else { Some(h) };
            self.first_dir[h] = None;
            self.next_swing(h, air || same);
            self.swung(h, air);
        } else if c.note0 <= k && k < c.arc0 {
            let j = k - c.note0;
            let h = j / 108;
            let (cell, d) = ((j / 9 % 12) as i64, (j % 9) as i64);
            let tick = self.tick.unwrap();
            self.loops.add(&c, tick, h as i64, cell, d, true);
            self.notes_now[h].push((j % 108) as i64);
            self.add_cell(h as u8, cell as u8);
            self.cell_tick[cell as usize] = Some(tick);
            self.col(h, cell);
            self.ahead(h, cell);
            self.set_rest(h, d);
            self.opened[h] = true;
            self.stack[h] += 1;
            if self.first_dir[h].is_none() {
                self.first_dir[h] = Some(d);
            }
            if d != 8 {
                self.swing_dir[h] = Some((d, tick));
            }
            self.need_note = None;
            self.note_ticks.push(tick);
            self.hit_now();
        } else if c.arc0 <= k && k < c.bomb0 {
            self.arc_done[k - c.arc0] = true;
            self.arc_ticks.push(self.tick.unwrap());
        } else if c.bomb0 <= k && k < c.wall0 {
            let cell = (k - c.bomb0) as i64;
            self.add_cell(2, cell as u8);
            self.bomb_tick[cell as usize] = self.tick;
            self.bomb(cell);
            self.objects_now = true;
        } else if c.wall0 <= k && k < c.dur0 {
            self.pending_wall = k - c.wall0;
            self.need_dur = true;
            self.objects_now = true;
        } else if c.dur0 <= k && k < c.events_end {
            self.add_wall(self.pending_wall, c.durations[k - c.dur0]);
            self.need_dur = false;
        }
    }

    fn off_times(&self, k: usize) -> bool {
        self.times.is_some()
            && self.c.time0 <= k
            && k < self.c.swing0
            && !self.in_times(Some(self.t0 + (k - self.c.time0) as i64))
    }
}

#[pyclass(module = "bsgen._rules")]
struct Rules {
    s: Arc<Mutex<State>>,
}

impl Rules {
    fn st(&self) -> std::sync::MutexGuard<'_, State> {
        self.s.lock().unwrap()
    }
}

#[pymethods]
impl Rules {
    #[new]
    #[pyo3(signature = (consts, tick, kind, hand, forehand, air, cell, dir, shape, dur, end_tick, swings, objects,
                        max_loop, behind, dry, turn, copy, hold, times, double_gap, cell_gap, swing_gap, uncross, near_gap,
                        wall_gap, vision_gap, slide_gap))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        consts: &NativeConsts,
        tick: PyReadonlyArray1<i64>,
        kind: PyReadonlyArray1<i8>,
        hand: PyReadonlyArray1<i8>,
        forehand: PyReadonlyArray1<bool>,
        air: PyReadonlyArray1<bool>,
        cell: PyReadonlyArray1<i8>,
        dir: PyReadonlyArray1<i8>,
        shape: PyReadonlyArray1<i16>,
        dur: PyReadonlyArray1<i32>,
        end_tick: i64,
        swings: bool,
        objects: i64,
        max_loop: i64,
        behind: i64,
        dry: f64,
        turn: f64,
        copy: Option<(i64, i64)>,
        hold: f64,
        times: Option<Vec<i64>>,
        double_gap: i64,
        cell_gap: i64,
        swing_gap: i64,
        uncross: bool,
        near_gap: i64,
        wall_gap: i64,
        vision_gap: i64,
        slide_gap: i64,
    ) -> PyResult<Self> {
        let c = consts.c.clone();
        let mut s = State {
            c: c.clone(),
            end: end_tick,
            swings,
            objects,
            turn,
            hold,
            times,
            behind,
            double_gap,
            double: None,
            swing_gap,
            slide_gap,
            uncross,
            cell_gap,
            cell_tick: [None; 12],
            bomb_tick: [None; 12],
            near_gap,
            wall_gap,
            vision_gap,
            front_tick: [None, None],
            cols: [None, None],
            moments_n: 0,
            doubles_n: 0,
            swings_n: 0,
            air_free_n: 0,
            head: [None, None],
            rest: [None, None],
            owe: [false, false],
            loops: Loops {
                limit: max_loop * c.ticks,
                dry,
                copy,
                sym: Vec::new(),
                ticks: Vec::new(),
                codes: Vec::new(),
                expect: Vec::new(),
                overused: Vec::new(),
                repeats: Vec::new(),
                stale: false,
            },
            tick: None,
            t0: 0,
            face: [None, None],
            need_note: None,
            need_dur: false,
            swing_tick: [None, None],
            air: [true, true],
            cells: Vec::new(),
            after_time: false,
            opened: [false, false],
            first_dir: [None, None],
            stack: [0, 0],
            prev_moment: [None, None],
            notes_now: [Vec::new(), Vec::new()],
            prev_note: [c.no_note, c.no_note],
            arc_done: [false, false],
            note_ticks: Vec::new(),
            arc_ticks: Vec::new(),
            objects_now: false,
            walls: Vec::new(),
            pending_wall: 0,
            need_time: false,
            swing_dir: [None, None],
            last_swing: [None, None],
            hit: None,
            run_gap: None,
            run_from: None,
        };
        let (tick, kind, hand, forehand, air) = (tick.as_array(), kind.as_array(), hand.as_array(), forehand.as_array(), air.as_array());
        let (cell, dir, shape, dur) = (cell.as_array(), dir.as_array(), shape.as_array(), dur.as_array());
        let [k_swing, k_note, k_arc, k_bomb, k_wall] = c.kinds;
        for i in 0..tick.len() {
            let t = tick[i];
            if s.tick != Some(t) {
                s.new_moment(t);
            }
            let (h, k) = (hand[i] as usize, kind[i] as i64);
            if k == k_note {
                let (cl, d) = (cell[i] as i64, dir[i] as i64);
                s.loops.add(&c, t, h as i64, cl, d, false);
                s.notes_now[h].push(cl * 9 + d);
                s.opened[h] = true;
                s.ahead(h, cl);
                if s.first_dir[h].is_none() {
                    s.first_dir[h] = Some(d);
                }
                if d != 8 {
                    s.swing_dir[h] = Some((d, t));
                }
                s.add_cell(h as u8, cl as u8);
                s.cell_tick[cl as usize] = Some(t);
                s.col(h, cl);
                s.set_rest(h, d);
                s.stack[h] += 1;
                s.note_ticks.push(t);
                s.hit_now();
            } else if k == k_arc {
                s.arc_ticks.push(t);
            } else if k == k_swing {
                let (fh, a) = (forehand[i], air[i]);
                let same = s.face[h] == Some(fh) && !a;
                s.face[h] = Some(fh);
                s.air[h] = a;
                s.swing_tick[h] = Some(t);
                if s.swing_tick[1 - h] == Some(t) {
                    s.double = Some(t);
                }
                s.first_dir[h] = None;
                s.next_swing(h, a || same);
                s.swung(h, a);
            } else if k == k_wall {
                s.add_wall(shape[i] as usize, dur[i] as i64);
            } else if k == k_bomb {
                s.bomb_tick[cell[i] as usize] = Some(t);
                s.bomb(cell[i] as i64);
            }
        }
        s.loops.check(&c);
        s.after_time = false;
        Ok(Rules { s: Arc::new(Mutex::new(s)) })
    }

    fn begin(&self, t0: i64, tick: i64) {
        let mut s = self.st();
        s.t0 = t0;
        s.new_moment(tick);
        s.need_time = true;
    }

    fn push(&self, k: usize) {
        self.st().push(k);
    }

    fn mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.st().mask())
    }

    fn penalties(&self) -> Vec<(usize, f64)> {
        self.st().penalties()
    }

    fn reset_merge(&self) -> Vec<(usize, usize)> {
        self.st().reset_merge()
    }

    fn way_on(&self) -> usize {
        self.st().way_on()
    }

    fn between(&self) -> bool {
        self.st().between()
    }

    fn off_times(&self, k: usize) -> bool {
        self.st().off_times(k)
    }

    fn snapshot(&self) -> Rules {
        Rules { s: Arc::new(Mutex::new(self.st().clone())) }
    }

    #[getter]
    fn tick(&self) -> Option<i64> {
        self.st().tick
    }

    #[getter]
    fn t0(&self) -> i64 {
        self.st().t0
    }

    #[getter]
    fn end(&self) -> i64 {
        self.st().end
    }

    #[getter]
    fn need_note(&self) -> Option<usize> {
        self.st().need_note
    }

    #[getter]
    fn need_dur(&self) -> bool {
        self.st().need_dur
    }

    #[getter]
    fn times(&self) -> Option<Vec<i64>> {
        self.st().times.clone()
    }

    #[getter]
    fn prev_note(&self) -> Vec<i64> {
        self.st().prev_note.to_vec()
    }
}

#[pyclass(module = "bsgen._rules")]
struct Step {
    n: usize,
    vocab: usize,
    allowed: Vec<bool>,
    pen: Vec<(usize, usize, f32)>,
    merges: Vec<Vec<(usize, usize)>>,
    way: Vec<Option<usize>>,
}

fn states(rules: &[PyRef<'_, Rules>]) -> Vec<Arc<Mutex<State>>> {
    rules.iter().map(|r| r.s.clone()).collect()
}

#[pyfunction]
fn prep(py: Python<'_>, rules: Vec<PyRef<'_, Rules>>, done: Vec<bool>, vocab: usize) -> Step {
    let st = states(&rules);
    drop(rules);
    py.detach(move || {
        let n = st.len();
        let mut step = Step { n, vocab, allowed: vec![false; n * vocab], pen: Vec::new(), merges: Vec::new(), way: Vec::new() };
        for (i, s) in st.iter().enumerate() {
            let mut s = s.lock().unwrap();
            let row = &mut step.allowed[i * vocab..(i + 1) * vocab];
            if done[i] {
                row[s.c.eos] = true;
            } else {
                let m = s.mask();
                let k = vocab.min(m.len());
                row[..k].copy_from_slice(&m[..k]);
                for (t, a) in s.penalties() {
                    step.pen.push((i, t, a as f32));
                }
            }
            step.merges.push(s.reset_merge());
            step.way.push(if !done[i] && s.times.is_some() && s.tick.is_some() { Some(s.way_on()) } else { None });
        }
        step
    })
}

fn rows<'a>(lg: &'a mut PyReadwriteArray2<'_, f32>, step: &Step) -> PyResult<&'a mut [f32]> {
    let shape = lg.shape();
    if shape != [step.n, step.vocab] {
        return Err(PyValueError::new_err(format!("logits {shape:?}, rules {}x{}", step.n, step.vocab)));
    }
    lg.as_slice_mut().map_err(|_| PyValueError::new_err("logits must be contiguous"))
}

#[pyfunction]
fn adjust(py: Python<'_>, step: &Step, mut lg: PyReadwriteArray2<'_, f32>, temp: f32, arc_boost: f32, dot_bias: f32,
          top_bias: f32, note0: usize, arc0: usize) -> PyResult<Vec<f64>> {
    let a = rows(&mut lg, step)?;
    Ok(py.detach(|| {
        let v = step.vocab;
        let mut bp = Vec::with_capacity(step.n);
        for i in 0..step.n {
            let row = &mut a[i * v..(i + 1) * v];
            let ok = &step.allowed[i * v..(i + 1) * v];
            let m = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x / temp));
            let (mut all, mut off) = (0.0f64, 0.0f64);
            for (x, &o) in row.iter().zip(ok) {
                let e = ((x / temp - m) as f64).exp();
                all += e;
                if !o {
                    off += e;
                }
            }
            bp.push(off / all);
            if arc_boost != 0.0 {
                for t in arc0..(arc0 + 2).min(v) {
                    row[t] += arc_boost;
                }
            }
            if dot_bias != 0.0 {
                for t in (note0 + 8..(note0 + 24 * 9).min(v)).step_by(9) {
                    row[t] -= dot_bias;
                }
            }
            if top_bias != 0.0 {
                for t in note0..(note0 + 24 * 9).min(v) {
                    if (t - note0) / 9 % 12 >= 8 {
                        row[t] -= top_bias;
                    }
                }
            }
        }
        for &(i, t, amt) in &step.pen {
            if t < v {
                a[i * v + t] -= amt;
            }
        }
        bp
    }))
}

fn logsumexp(x: impl Iterator<Item = f32> + Clone) -> f32 {
    let m = x.clone().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        return m;
    }
    m + x.map(|v| (v - m).exp()).sum::<f32>().ln()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn finish(py: Python<'_>, step: &Step, mut lg: PyReadwriteArray2<'_, f32>, temp: f32, top_p: f32, min_p: f32,
          u: Vec<f32>, xtc: Vec<bool>, xtc_threshold: f32, xtc_floor: f32, time0: usize, swing0: usize, eos: usize)
          -> PyResult<Vec<usize>> {
    let a = rows(&mut lg, step)?;
    py.detach(|| {
        let v = step.vocab;
        let mut out = Vec::with_capacity(step.n);
        for i in 0..step.n {
            let row = &mut a[i * v..(i + 1) * v];
            let ok = &step.allowed[i * v..(i + 1) * v];
            for &(to, fr) in &step.merges[i] {
                row[to] = logsumexp([row[to], row[fr]].into_iter());
            }
            if let Some(way) = step.way[i]
                && ok[way]
            {
                let shut = (time0..swing0).chain([eos]).filter(|&t| !ok[t]).map(|t| row[t]);
                row[way] = logsumexp(std::iter::once(row[way]).chain(shut.collect::<Vec<_>>()));
            }
            let mut cand: Vec<(f32, usize)> = (0..v).filter(|&t| ok[t]).map(|t| (row[t] / temp, t)).collect();
            if cand.is_empty() {
                return Err(PyRuntimeError::new_err(format!("row {i}: the rules allow no token")));
            }
            let m = cand.iter().fold(f32::NEG_INFINITY, |m, c| m.max(c.0));
            let mut sum = 0.0f32;
            for c in cand.iter_mut() {
                c.0 = (c.0 - m).exp();
                sum += c.0;
            }
            for c in cand.iter_mut() {
                c.0 /= sum;
            }
            if xtc.get(i).copied().unwrap_or(false) {
                let top: Vec<usize> = (0..cand.len()).filter(|&j| cand[j].0 >= xtc_threshold).collect();
                if top.len() >= 2 {
                    let least = top.iter().map(|&j| cand[j].0).fold(f32::INFINITY, f32::min);
                    for &j in &top {
                        if cand[j].0 > least {
                            cand[j].0 = 0.0;
                        }
                    }
                    for c in cand.iter_mut() {
                        if c.0 < xtc_floor {
                            c.0 = 0.0;
                        }
                    }
                }
            }
            cand.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
            if min_p > 0.0 {
                let floor = min_p * cand[0].0;
                for c in cand.iter_mut() {
                    if c.0 < floor {
                        c.0 = 0.0;
                    }
                }
            } else {
                let mut cum = 0.0f32;
                for c in cand.iter_mut() {
                    cum += c.0;
                    if cum - c.0 >= top_p {
                        c.0 = 0.0;
                    }
                }
            }
            let total: f32 = cand.iter().map(|c| c.0).sum();
            let target = u[i] * total;
            let mut cum = 0.0f32;
            let mut pick = None;
            for c in &cand {
                if c.0 > 0.0 {
                    cum += c.0;
                    pick = Some(c.1);
                    if cum > target {
                        break;
                    }
                }
            }
            out.push(pick.unwrap_or(cand[0].1));
        }
        Ok(out)
    })
}

#[pymodule]
fn _rules(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeConsts>()?;
    m.add_class::<Rules>()?;
    m.add_class::<Step>()?;
    m.add_function(wrap_pyfunction!(prep, m)?)?;
    m.add_function(wrap_pyfunction!(adjust, m)?)?;
    m.add_function(wrap_pyfunction!(finish, m)?)?;
    Ok(())
}
