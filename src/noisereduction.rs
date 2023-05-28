/*
#define ANR_DLINE_SIZE 512
#define ANR_MASK (ANR_DLINE_SIZE - 1)

struct wdsp_ANR {
    int in_idx;
    f32_t d[ANR_DLINE_SIZE];
    f32_t w[ANR_DLINE_SIZE];

    int taps;  // 16..128, >= delay
    int delay; // 2..128, <= taps
    int position;
    f32_t two_mu; // gain, 1e-7..8.192e-2, step *= 2 (20 steps)
    f32_t gamma;  // leakage, 1e-3..8192.0, step *= 2 (23 steps)
    f32_t lidx;
    f32_t lidx_min;
    f32_t lidx_max;
    f32_t ngamma;
    f32_t den_mult;
    f32_t lincr;
    f32_t ldecr;
};

typedef struct wdsp_ANR wdsp_ANR_t;

wdsp_ANR_t* wdsp_anr_create(int nr_type, int NR_TAPS, int NR_DLY, float NR_GAIN,
        float NR_LEAKAGE) {
    wdsp_ANR_t *w = (wdsp_ANR_t *)malloc(sizeof(wdsp_ANR_t));
    memset(w, 0, sizeof(wdsp_ANR_t));

    w->taps = NR_TAPS; //(int) nr_param[NR_TAPS];
    w->delay = NR_DLY; //(int) nr_param[NR_DLY];
    w->position = 0;
    w->two_mu = NR_GAIN;   // nr_param[NR_GAIN];
    w->gamma = NR_LEAKAGE; // nr_param[NR_LEAKAGE];
    w->lidx = 120.0;
    w->lidx_min = 120.0;
    w->lidx_max = 200.0;
    w->ngamma = 0.001;
    w->den_mult = 6.25e-10;
    w->lincr = 1.0;
    w->ldecr = 3.0;
    // printf("wdsp_ANR_init type=%d taps=%d delay=%d gain=%.9f
    // leakage=%.9f\n",
    //    nr_type, w->taps, w->delay, w->two_mu, w->gamma);
    return w;
}

void wdsp_anr_filter(wdsp_ANR_t *w, int nr_type, int ns_out, float* in, float* out) {
    int idx;
    f32_t c0, c1;
    f32_t y, error, sigma, inv_sigp;
    f32_t nel, nev;
    f32_t out_f;

    for (int i = 0; i < ns_out; i++) {
        w->d[w->in_idx] = in[i];
        y = 0;
        sigma = 0;

        for (int j = 0; j < w->taps; j++) {
            idx = (w->in_idx + j + w->delay) & ANR_MASK;
            y += w->w[j] * w->d[idx];
            sigma += w->d[idx] * w->d[idx];
        }

        inv_sigp = 1.0 / (sigma + 1e-10);
        error = w->d[w->in_idx] - y;

        if (nr_type == 0)
            out_f = error; // notch filter
        else
            out_f = y * 4.0; // noise reduction

        out[i] = out_f;

        if ((nel = error * (1.0 - w->two_mu * sigma * inv_sigp)) < 0.0)
            nel = -nel;

        if ((nev = w->d[w->in_idx] - (1.0 - w->two_mu * w->ngamma) * y -
                    w->two_mu * error * sigma * inv_sigp) < 0.0)
            nev = -nev;

        if (nev < nel) {
            if ((w->lidx += w->lincr) > w->lidx_max)
                w->lidx = w->lidx_max;
            else if ((w->lidx -= w->ldecr) < w->lidx_min)
                w->lidx = w->lidx_min;
        }

        w->ngamma = w->gamma * (w->lidx * w->lidx) * (w->lidx * w->lidx) *
                    w->den_mult;

        c0 = 1.0 - w->two_mu * w->ngamma;
        c1 = w->two_mu * error * inv_sigp;

        for (int j = 0; j < w->taps; j++) {
            idx = (w->in_idx + j + w->delay) & ANR_MASK;
            w->w[j] = c0 * w->w[j] + c1 * w->d[idx];
        }

        w->in_idx = (w->in_idx + ANR_MASK) & ANR_MASK;
    }
}

void wdsp_anr_destroy(wdsp_ANR_t * w) {
    free(w);
}

*/

// Convert the above c code into idiomatic rust code
// Use vectors and avoid unsafe accesses

use std::collections::VecDeque;

pub struct NoiseReduction {
    nr_type: NoiseReductionType,
    d: VecDeque<f32>,
    w: Vec<f32>,
    taps: usize,
    delay: usize,
    two_mu: f32,
    gamma: f32,
    lidx: f32,
    lidx_min: f32,
    lidx_max: f32,
    ngamma: f32,
    den_mult: f32,
    lincr: f32,
    ldecr: f32,
}

#[derive(PartialEq)]
pub enum NoiseReductionType {
    Notch,
    NoiseReduction,
}

impl NoiseReduction {
    pub fn new(
        nr_type: NoiseReductionType,
        nr_taps: usize,
        nr_dly: usize,
        nr_gain: f32,
        nr_leakage: f32,
    ) -> Self {
        let mut d = VecDeque::with_capacity(nr_taps + nr_dly);
        for _ in 0..(nr_taps + nr_dly) {
            d.push_back(0.0);
        }
        Self {
            nr_type,
            d,
            w: vec![0.0; nr_taps],
            taps: nr_taps,
            delay: nr_dly,
            two_mu: nr_gain,
            gamma: nr_leakage,
            lidx: 120.0,
            lidx_min: 120.0,
            lidx_max: 200.0,
            ngamma: 0.001,
            den_mult: 6.25e-10,
            lincr: 1.0,
            ldecr: 3.0,
        }
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        let mut idx: usize;
        let mut c0: f32;
        let mut c1: f32;
        let mut y: f32;
        let mut error: f32;
        let mut sigma: f32;
        let mut inv_sigp: f32;
        let mut nel: f32;
        let mut nev: f32;
        let mut out_f: f32;

        let ns_out = buf.len();
        for i in 0..ns_out {
            self.d.push_front(buf[i]);
            y = 0.0;
            sigma = 0.0;

            for j in 0..self.taps {
                idx = j + self.delay;
                y += self.w[j] * self.d[idx];
                sigma += self.d[idx] * self.d[idx];
            }

            inv_sigp = 1.0 / (sigma + 1e-10);
            error = self.d.front().unwrap() - y;

            out_f = 0.0;
            if self.nr_type == NoiseReductionType::Notch {
                out_f = error; // notch filter
            } else if self.nr_type == NoiseReductionType::NoiseReduction {
                out_f = y * 4.0; // noise reduction
            }

            buf[i] = out_f;

            nel = error * (1.0 - self.two_mu * sigma * inv_sigp);
            if nel < 0.0 {
                nel = -nel;
            }

            nev = self.d.front().unwrap() - (1.0 - self.two_mu * self.ngamma);
            if (nev * y - self.two_mu * error * sigma * inv_sigp) < 0.0 {
                nev = -nev;
            }

            if nev < nel {
                self.lidx += self.lincr;
                if self.lidx > self.lidx_max {
                    self.lidx = self.lidx_max;
                } else {
                    self.lidx -= self.ldecr;
                    if self.lidx < self.lidx_min {
                        self.lidx = self.lidx_min;
                    }
                }
            }

            self.ngamma =
                self.gamma * (self.lidx * self.lidx) * (self.lidx * self.lidx) * self.den_mult;

            c0 = 1.0 - self.two_mu * self.ngamma;

            c1 = self.two_mu * error * inv_sigp;

            for j in 0..self.taps {
                idx = j + self.delay;
                self.w[j] = c0 * self.w[j] + c1 * self.d[idx];
            }
        }
    }
}
