/*

struct nb_Wild_t {
    f32_t thresh;
    s1_t taps;
    s1_t impulse_samples;

    #define MAX_ORDER 40    // aka taps
    #define MAX_IMPULSE_LEN 41
    #define MAX_PL ((MAX_IMPULSE_LEN - 1) / 2)

    #define WORKING_BUFFER
    #ifdef WORKING_BUFFER
        #define DIM_WBUF (FASTFIR_OUTBUF_SIZE + MAX_ORDER*2 + MAX_PL*2)
        f32_t working_buffer[DIM_WBUF];
    #else
        f32_t last_frame_end[MAX_ORDER + MAX_PL]; // this takes the last samples from the previous frame to do the prediction within the boundaries
    #endif
};
*/

use futuredsp::{fir::NonResamplingFirKernel, UnaryKernel};

pub struct NoiseBlankerWild {
    thresh: f32,
    taps: usize,
    impulse_samples: usize,

    working_buffer: Vec<f32>,
}

/*
static nb_Wild_t nb_Wild[MAX_RX_CHANS];

void nb_Wild_init(int rx_chan, TYPEREAL nb_param[NOISE_PARAMS])
{
    nb_Wild_t *w = &nb_Wild[rx_chan];
    memset(w, 0, sizeof(nb_Wild_t));
    w->thresh = nb_param[NB_THRESH];
    w->taps = (s1_t) nb_param[NB_TAPS];
    w->impulse_samples = (s1_t) nb_param[NB_SAMPLES];
    //printf("nb_Wild_init: thresh=%.1e taps=%d samples=%d\n", w->thresh, w->taps, w->impulse_samples);
}
*/

impl NoiseBlankerWild {
    pub fn new(NB_THRESH: f32, NB_TAPS: u32, NB_SAMPLES: u32) -> Self {
        Self {
            thresh: NB_THRESH,
            taps: NB_TAPS as usize,
            impulse_samples: NB_SAMPLES as usize,
            working_buffer: Vec::new(),
        }
    }

    // Convert the above c code into idiomatic rust code
    // Use vectors and avoid unsafe accesses

    /*

    void nb_Wild_inner(int rx_chan, int nsamps, f32_t *samps)
    {
        int i;
        nb_Wild_t *w = &nb_Wild[rx_chan];
        //printf("nb_Wild_inner: thresh=%.2f taps=%d samples=%d\n", w->thresh, w->taps, w->impulse_samples);

        u4_t impulse_length = w->impulse_samples | 1;   // must be odd
        u4_t PL = (impulse_length - 1) / 2;     // has to be (impulse_length-1)/2
        int order = w->taps;    // lpc's order
        f32_t lpcs[order + 1];      // we reserve one more than "order" because of a leading "1"
        f32_t reverse_lpcs[order + 1];      // this takes the reversed order lpc coefficients
        f32_t firStateF32[nsamps + order];
        f32_t tempsamp[nsamps];
        f32_t sigma2;   // taking the variance of the input
        f32_t lpc_power;
        #define N_IMPULSE_COUNT 20
        int impulse_positions[N_IMPULSE_COUNT];
        int search_pos = 0;
        int impulse_count = 0;

        f32_t R[order + 1];  // takes the autocorrelation results
        f32_t k, alfa;

        f32_t any[order + 1]; // some internal buffers for the levinson durben algorithm

        f32_t Wfw[impulse_length], Wbw[impulse_length]; // taking linear windows for the combination of fwd and bwd

        f32_t s;

        #ifdef WORKING_BUFFER
            // copy incomming samples to the end of our working buffer
            memcpy(&w->working_buffer[2*PL + 2*order], samps, nsamps * sizeof(f32_t));
        #endif
    */

    pub fn process(&mut self, samps: &mut [f32]) {
        let impulse_length = self.impulse_samples | 1; // must be odd
        let PL = (impulse_length - 1) / 2; // has to be (impulse_length-1)/2
        let order = self.taps; // lpc's order
        let mut lpcs = vec![0.0; order + 1]; // we reserve one more than "order" because of a leading "1"
        let mut reverse_lpcs = vec![0.0; order + 1]; // this takes the reversed order lpc coefficients
        let mut tempsamp = vec![0.0; samps.len()];
        //let mut sigma2; // taking the variance of the input
        //let mut lpc_power;
        let N_IMPULSE_COUNT = 20;
        let mut impulse_positions = vec![0; N_IMPULSE_COUNT];
        let mut search_pos = 0;
        let mut impulse_count = 0;

        let mut R = vec![0.0; order + 1]; // takes the autocorrelation results
        let mut k;
        let mut alfa;

        let mut any = vec![0.0; order + 1]; // some internal buffers for the levinson durben algorithm

        let mut Wfw = vec![0.0; impulse_length]; // taking linear windows for the combination of fwd and bwd
        let mut Wbw = vec![0.0; impulse_length]; // taking linear windows for the combination of fwd and bwd

        let mut s;

        // copy incomming samples to the end of our working buffer
        //memcpy(&w->working_buffer[2*PL + 2*order], samps, nsamps * sizeof(f32_t));
        self.working_buffer.resize(2 * PL + 2 * order + samps.len(), 0.0);
        for i in 0..samps.len() {
            self.working_buffer[2 * PL + 2 * order + i] = samps[i];
        }

        /*
            // generating 2 windows for the combination of the 2 predictors
            // will be a constant window later
            for (i = 0; i < impulse_length; i++) {
                Wbw[i] = 1.0 * i / (impulse_length - 1);
                Wfw[impulse_length - i - 1] = Wbw[i];
            }

            // calculate the autocorrelation of samps (moving by max. of #order# samples)
            for (i = 0; i < (order + 1); i++) {
                 // R is carrying the cross-correlations
                #ifdef WORKING_BUFFER
                    arm_dot_prod_f32(&w->working_buffer[order + PL], &w->working_buffer[order + PL + i], nsamps - i, &R[i]);
                #else
                    arm_dot_prod_f32(&samps[0], &samps[i], nsamps - i, &R[i]);
                #endif
            }

            // alternative Levinson Durben algorithm to calculate the lpc coefficients from the cross-correlation
            R[0] = R[0] * (1.0 + 1.0e-9);
            lpcs[0] = 1;   //set lpc 0 to 1

            for (i = 1; i < order + 1; i++) {
                lpcs[i] = 0;    // fill rest of array with zeros - could be done by memfill
            }

            alfa = R[0];

            for (int m = 1; m <= order; m++) {
                s = 0.0;
                for (int u = 1; u < m; u++) {
                    assert_array_dim(m-u, order + 1);
                    s = s + lpcs[u] * R[m - u];
                }

                k = -(R[m] + s) / alfa;

                for (int v = 1; v < m; v++)
                    any[v] = lpcs[v] + k * lpcs[m - v];

                for (int w = 1; w < m; w++)
                lpcs[w] = any[w];

                lpcs[m] = k;
                alfa = alfa * (1 - k * k);
            }
            // end of Levinson Durben algorithm

            for (int o = 0; o < order + 1; o++)     // store the reverse order coefficients separately
                reverse_lpcs[order - o] = lpcs[o];  // for the matched impulse filter
        */
        for i in 0..impulse_length {
            Wbw[i] = 1.0 * (i as f32) / ((impulse_length as f32) - 1.0);
            Wfw[impulse_length - i - 1] = Wbw[i];
        }

        // calculate the autocorrelation of samps (moving by max. of #order# samples)
        for i in 0..(order + 1) {
            // R is carrying the cross-correlations
            for j in 0..(samps.len() - i) {
                R[i] +=
                    self.working_buffer[order + PL + j] * self.working_buffer[order + PL + j + i];
            }
        }

        // alternative Levinson Durben algorithm to calculate the lpc coefficients from the cross-correlation
        R[0] = R[0] * (1.0 + 1.0e-9);
        lpcs[0] = 1.0; //set lpc 0 to 1

        for i in 1..(order + 1) {
            lpcs[i] = 0.0; // fill rest of array with zeros - could be done by memfill
        }

        alfa = R[0];

        for m in 1..(order + 1) {
            s = 0.0;
            for u in 1..m {
                s += lpcs[u] * R[m - u];
            }

            k = -(R[m] + s) / alfa;

            for v in 1..m {
                any[v] = lpcs[v] + k * lpcs[m - v];
            }

            for w in 1..m {
                lpcs[w] = any[w];
            }

            lpcs[m] = k;
            alfa = alfa * (1.0 - k * k);
        }

        // end of Levinson Durben algorithm

        for o in 0..(order + 1) {
            // store the reverse order coefficients separately
            reverse_lpcs[order - o] = lpcs[o]; // for the matched impulse filter
        }
        /*
            arm_fir_instance_f32 LPC;
            arm_fir_init_f32(&LPC, order + 1, &reverse_lpcs[0], &firStateF32[0], nsamps);    // we are using the same function as used in freedv

            // do the inverse filtering to eliminate voice and enhance the impulses
            #ifdef WORKING_BUFFER
                arm_fir_f32(&LPC, &w->working_buffer[order + PL], tempsamp, nsamps);
            #else
                arm_fir_f32(&LPC, samps, tempsamp, nsamps);
            #endif

            arm_fir_init_f32(&LPC, order + 1, &lpcs[0], &firStateF32[0], nsamps);            // we are using the same function as used in freedv
            arm_fir_f32(&LPC, tempsamp, tempsamp, nsamps);  // do a matched filtering to detect an impulse in our now voiceless signal

            arm_var_f32(tempsamp, nsamps, &sigma2);  // calculate sigma2 of the original signal ? or tempsignal
            arm_power_f32(lpcs, order, &lpc_power);     // calculate the sum of the squares (the "power") of the lpc's
        */
        let LPC = NonResamplingFirKernel::<f32, f32, _, _>::new(reverse_lpcs.clone());
        LPC.work(&mut self.working_buffer[order + PL..], &mut tempsamp[..]);

        let LPC_l = NonResamplingFirKernel::<f32, f32, _, _>::new(lpcs.clone());
        let mut tempsampout = vec![0.0; tempsamp.len()];
        LPC_l.work(&mut tempsamp[..], &mut tempsampout[..]);
        for i in 0..tempsamp.len() {
            tempsamp[i] = tempsampout[i];
        }

        // sigma2 is variance of tempsamp
        let mean2 = tempsamp.iter().sum::<f32>() / tempsamp.len() as f32;
        let sigma2 =
            tempsamp.iter().map(|x| (x - mean2).powi(2)).sum::<f32>() / tempsamp.len() as f32;

        // lpc_power is sum of squares of lpcs
        let lpc_power = lpcs.iter().map(|x| x.powi(2)).sum::<f32>();
        /*
            f32_t impulse_threshold = w->thresh * sqrtf(sigma2 * lpc_power);  // set a detection level (3 is not really a final setting)

            search_pos = order + PL; // lower boundary problem has been solved! - so here we start from 1 or 0?
            impulse_count = 0;

            do {    // going through the filtered samples to find an impulse larger than the threshold

                if ((tempsamp[search_pos] > impulse_threshold) || (tempsamp[search_pos] < (-impulse_threshold))) {
                    impulse_positions[impulse_count] = search_pos - order; // save the impulse positions and correct it by the filter delay
                    impulse_count++;
                    search_pos += PL;   // set search_pos a bit away, cause we are already repairing this area later
                                        // and the next impulse should not be that close
                    }

                search_pos++;

            } while ((search_pos < nsamps) && (impulse_count < N_IMPULSE_COUNT)); // avoid upper boundary
        */
        let impulse_threshold = self.thresh * (sigma2 * lpc_power).sqrt();

        search_pos = order + PL;
        impulse_count = 0;

        loop {
            if tempsamp[search_pos].abs() > impulse_threshold {
                impulse_positions[impulse_count] = search_pos - order;
                impulse_count += 1;
                search_pos += PL;
            }

            search_pos += 1;

            if (search_pos >= samps.len()) || (impulse_count >= N_IMPULSE_COUNT) {
                break;
            }
        }
        /*
            // boundary handling has to be fixed later
            // as a result we now will not find any impulse in these areas

            // from here: reconstruction of the impulse-distorted audio part:

            // first we form the forward and backward prediction transfer functions from the lpcs
            // that is easy, as they are just the negated coefficients without the leading "1"
            // we can do this in place of the lpcs, as they are not used here anymore and being recalculated in the next frame

            arm_negate_f32(&lpcs[1], &lpcs[1], order);
            arm_negate_f32(&reverse_lpcs[0], &reverse_lpcs[0], order);

            f32_t Rfw[impulse_length + order]; // takes the forward predicted audio restoration
            f32_t Rbw[impulse_length + order]; // takes the backward predicted audio restoration

            for (int j = 0; j < impulse_count; j++) {
                // we have to copy some samples from the original signal as
                // basis for the reconstructions - could be done by memcopy
                for (int k = 0; k < order; k++) {
                    #ifdef WORKING_BUFFER
                        i = impulse_positions[j] + k;
                        assert_array_dim(i, DIM_WBUF);
                        Rfw[k] = w->working_buffer[i];
                        i = order + PL + impulse_positions[j] + PL + k + 1;
                        assert_array_dim(i, DIM_WBUF);
                        Rbw[impulse_length+k] = w->working_buffer[i];
                    #else
                        i = impulse_positions[j] - PL - order + k;
                        if (i < 0) {  // this solves the prediction problem at the left boundary
                            i = impulse_positions[j] + k;
                            assert_array_dim(i, order + PL);
                            Rfw[k] = w->last_frame_end[i]; // take the sample from the last frame
                        } else {
                            assert_array_dim(i, nsamps);
                            Rfw[k] = samps[i]; //take the sample from this frame as we are away from the boundary
                        }

                        i = impulse_positions[j] + PL + k + 1;
                        assert_array_dim(i, nsamps);
                        Rbw[impulse_length + k] = samps[i];
                    #endif
                }

                for (i = 0; i < impulse_length; i++) {  // now we calculate the forward and backward predictions
                    arm_dot_prod_f32(&reverse_lpcs[0], &Rfw[i], order, &Rfw[i + order]);
                    arm_dot_prod_f32(&lpcs[1], &Rbw[impulse_length - i], order, &Rbw[impulse_length - i - 1]);
                }

                arm_mult_f32(&Wfw[0], &Rfw[order], &Rfw[order], impulse_length); // do the windowing, or better: weighting
                arm_mult_f32(&Wbw[0], &Rbw[0], &Rbw[0], impulse_length);

                // finally add the two weighted predictions and insert them into the original signal - thereby eliminating the distortion
                #ifdef WORKING_BUFFER
                    arm_add_f32(&Rfw[order], &Rbw[0], &w->working_buffer[order + impulse_positions[j]], impulse_length);
                #else
                    arm_add_f32(&Rfw[order], &Rbw[0], &samps[impulse_positions[j] - PL], impulse_length);
                #endif
            }

        */
        for i in 1..(order + 1) {
            lpcs[i] = -lpcs[i];
        }
        for i in 0..order {
            reverse_lpcs[i] = -reverse_lpcs[i];
        }

        let mut Rfw = vec![0.0; impulse_length + order];
        let mut Rbw = vec![0.0; impulse_length + order];

        // WORKING_BUFFER is defined

        for j in 0..impulse_count {
            for k in 0..order {
                let mut i = impulse_positions[j] + k;
                Rfw[k] = self.working_buffer[i];
                i = order + PL + impulse_positions[j] + PL + k + 1;
                Rbw[impulse_length + k] = self.working_buffer[i];
            }

            for i in 0..impulse_length {
                Rfw[i + order] = reverse_lpcs[0..order]
                    .iter()
                    .zip(&Rfw[i..])
                    .map(|(a, b)| a * b)
                    .sum();
                Rbw[impulse_length - i - 1] = lpcs[1..(order + 1)]
                    .iter()
                    .zip(&Rbw[impulse_length - i..])
                    .map(|(a, b)| a * b)
                    .sum();
            }

            for i in 0..impulse_length {
                Rfw[i + order] *= Wfw[i];
                Rbw[i] *= Wbw[i];
            }

            for i in 0..impulse_length {
                self.working_buffer[order + impulse_positions[j] + i] = Rfw[order + i] + Rbw[i];
            }
        }

        /*
            #ifdef WORKING_BUFFER
                // copy the samples of the current frame back to the samps buffer for output
                memcpy(samps, &w->working_buffer[order + PL], nsamps * sizeof(f32_t));

                // samples for next frame
                memcpy(w->working_buffer, &w->working_buffer[nsamps], (2*order + 2*PL) * sizeof(f32_t));
            #else
                for (int p = 0; p < (order + PL); p++) {
                    i = nsamps - 1 - order - PL + p;
                    assert_array_dim(i, nsamps);
                    assert_array_dim(p, order + PL);
                    w->last_frame_end[p] = samps[i];   // store samples from the current frame to use at the next frame
                }
            #endif
        }
        */

        // WORKING_BUFFER is defined

        for i in 0..samps.len() {
            samps[i] = self.working_buffer[order + PL + i];
        }

        for i in 0..(2 * order + 2 * PL) {
            self.working_buffer[i] = self.working_buffer[samps.len() + i];
        }
    }
}
