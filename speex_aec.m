%    Copyright (C) 2012      Waves Audio LTD
%    Copyright (C) 2003-2008 Jean-Marc Valin
%
%    File: speex_mdf.m
%    Echo canceller based on the MDF algorithm (see below)
% 
%    Redistribution and use in source and binary forms, with or without
%    modification, are permitted provided that the following conditions are
%    met:
% 
%    1. Redistributions of source code must retain the above copyright notice,
%    this list of conditions and the following disclaimer.
% 
%    2. Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in the
%    documentation and/or other materials provided with the distribution.
% 
%    3. The name of the author may not be used to endorse or promote products
%    derived from this software without specific prior written permission.
% 
%    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
%    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
%    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
%    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
%    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
				%    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
%    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
%    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
%    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%    POSSIBILITY OF SUCH DAMAGE.
%
%    Notes from original mdf.c:
%
%    The echo canceller is based on the MDF algorithm described in:
% 
%    J. S. Soo, K. K. Pang Multidelay block frequency adaptive filter, 
%    IEEE Trans. Acoust. Speech Signal Process., Vol. ASSP-38, No. 2, 
%    February 1990.
%    
%    We use the Alternatively Updated MDF (AUMDF) variant. Robustness to 
%    double-talk is achieved using a variable learning rate as described in:
%    
%    Valin, J.-M., On Adjusting the Learning Rate in Frequency Domain Echo 
%    Cancellation With Double-Talk. IEEE Transactions on Audio,
%    Speech and Language Processing, Vol. 15, No. 3, pp. 1030-1034, 2007.
%    http://people.xiph.org/~jm/papers/valin_taslp2006.pdf
%    
%    There is no explicit double-talk detection, but a continuous variation
%    in the learning rate based on residual echo, double-talk and background
%    noise.
%    
%    Another kludge that seems to work good: when performing the weight
%    update, we only move half the way toward the "goal" this seems to
%    reduce the effect of quantization noise in the update phase. This
%    can be seen as applying a gradient descent on a "soft constraint"
%    instead of having a hard constraint.
%    
%    Notes for this file:
%
%    Usage: 
%
%       speex_mdf_out = speex_mdf(Fs, u, d, filter_length, frame_size, dbg_var_name);
%       
%       Fs                  sample rate
%       u                   speaker signal, column vector in range [-1; 1]
%       d                   microphone signal, column vector in range [-1; 1]
%       filter_length       typically 250ms, i.e. 4096 @ 16k FS 
%                           must be a power of 2
%       frame_size          typically 8ms, i.e. 128 @ 16k Fs 
%                           must be a power of 2
%       dbg_var_name        internal state variable name to trace. 
%                           Default: 'st.leak_estimate'.
%
%    Jonathan Rouach <jonr@waves.com>
%    

function  speex_mdf_out = speex_mdf(Fs, u, d, filter_length, frame_size, dbg_var_name)

		fprintf('Starting Speex MDF (PBFDAF) algorithm.\n');

		st = speex_echo_state_init_mc_mdf(frame_size, filter_length, 1, 1, Fs);

		% which variable to trace
		if nargin<6
  		dbg_var_name = 'st.leak_estimate';
		end
		dbg = init_dbg(st, length(u));

		[e, dbg] = main_loop(st, float_to_short(u), float_to_short(d), dbg);

		speex_mdf_out.e = e/32768.0;
		speex_mdf_out.var1 = dbg.var1;

function x = float_to_short(x)
		x = x*32768.0;
		x(x< -32767.5) = -32768;
		x(x>  32766.5) =  32767;
		x = floor(0.5+x);
		end

function [e, dbg] = main_loop(st, u, d, dbg)

		e = zeros(size(u));
		y = zeros(size(u));

		% prepare waitbar
		try h_wb = waitbar(0, 'Processing...'); catch; end
		end_point = length(u);

		for n = 1:st.frame_size:end_point
	  	nStep = floor(n/st.frame_size)+1;

    if mod(nStep, 128)==0 && update_waitbar_check_wasclosed(h_wb, n, end_point, st.sampling_rate)
		  break;
		end

		u_frame = u(n:n+st.frame_size-1);
		d_frame = d(n:n+st.frame_size-1);

		[out, st] = speex_echo_cancellation_mdf(st, d_frame, u_frame);

		e(n:n+st.frame_size-1) = out*2;
		y(n:n+st.frame_size-1) = d_frame - out;
		dbg.var1(:, nStep) = reshape( eval(dbg_var_name),  numel(eval(dbg_var_name)), 1);

		end

		try close(h_wb); catch; end

		end
function st = speex_echo_state_init_mc_mdf(frame_size, filter_length, nb_mic, nb_speakers, sample_rate)

		st.K = nb_speakers;
		st.C = nb_mic;
		C=st.C;
		K=st.K;

		st.frame_size = frame_size;
		st.window_size = 2*frame_size;
		N = st.window_size;
		st.M = fix((filter_length+st.frame_size-1)/frame_size);
		M = st.M;
		st.cancel_count=0;
		st.sum_adapt = 0;
		st.saturated = 0;
		st.screwed_up = 0;

		%    /* This is the default sampling rate */
		st.sampling_rate = sample_rate;
		st.spec_average = (st.frame_size)/( st.sampling_rate);
		st.beta0 = (2.0*st.frame_size)/st.sampling_rate;
		st.beta_max = (.5*st.frame_size)/st.sampling_rate;
		st.leak_estimate = 0;

		st.e = zeros(N, C);
		st.x = zeros(N, K);
		st.input = zeros(st.frame_size, C);
		st.y = zeros(N, C);
		st.last_y = zeros(N, C);
		st.Yf = zeros(st.frame_size+1, 1);
		st.Rf = zeros(st.frame_size+1, 1);
		st.Xf = zeros(st.frame_size+1, 1);
		st.Yh = zeros(st.frame_size+1, 1);
		st.Eh = zeros(st.frame_size+1, 1);

		st.X = zeros(N, K, M+1);
		st.Y = zeros(N, C);
		st.E = zeros(N, C);
		st.W = zeros(N, K, M, C);
		st.foreground = zeros(N, K, M, C);
		st.PHI = zeros(frame_size+1, 1);
		st.power = zeros(frame_size+1, 1);
		st.power_1 = ones((frame_size+1), 1);
		st.window = zeros(N, 1);
		st.prop = zeros(M, 1);
		st.wtmp = zeros(N, 1);

		st.window = .5-.5*cos(2*pi*((1:N)'-1)/N);

		% /* Ratio of ~10 between adaptation rate of first and last block */
		decay = exp(-1/M);
		st.prop(1, 1) = .7;
		for i=2:M
		  st.prop(i, 1) = st.prop(i-1, 1) * decay;
		end

		st.prop = (.8 * st.prop)./sum(st.prop);

		st.memX = zeros(K, 1);
		st.memD = zeros(C, 1);
		st.memE = zeros(C, 1);
		st.preemph = .98;
    if (st.sampling_rate<12000)
		    st.notch_radius = .9;
    elseif (st.sampling_rate<24000)
		    st.notch_radius = .982;
		else
		    st.notch_radius = .992;
		end

		st.notch_mem = zeros(2*C, 1);
		st.adapted = 0;
		st.Pey = 1;
		st.Pyy = 1;

		st.Davg1 = 0; st.Davg2 = 0;
		st.Dvar1 = 0; st.Dvar2 = 0;
		end

function dbg = init_dbg(st, len)
		dbg.var1 = zeros(numel(eval(dbg_var_name)), fix(len/st.frame_size));
		end

function [out, st] = speex_echo_cancellation_mdf(st, in, far_end)

		N = st.window_size;
		M = st.M;
		C = st.C;
		K = st.K;

		Pey_cur = 1;
		Pyy_cur = 1;

		out = zeros(st.frame_size, C);

		st.cancel_count = st.cancel_count + 1;

		%ss=.35/M;
		ss = 0.5/M;
		ss_1 = 1-ss;

		for chan = 1:C
		% Apply a notch filter to make sure DC doesn't end up causing problems
		[st.input(:, chan), st.notch_mem(:, chan)] = filter_dc_notch16(in(:, chan), st.notch_radius, st.frame_size, st.notch_mem(:, chan));
		% Copy input data to buffer and apply pre-emphasis
		  for i=1:st.frame_size
		    tmp32 = st.input(i, chan)- (st.preemph* st.memD(chan));
		    st.memD(chan) = st.input(i, chan);
		    st.input(i, chan) = tmp32;
		  end
		end

		for speak = 1:K
		  for i =1:st.frame_size
		    st.x(i, speak) = st.x(i+st.frame_size, speak);
		    tmp32 = far_end(i, speak) - st.preemph * st.memX(speak);
		    st.x(i+st.frame_size, speak) = tmp32;
		    st.memX(speak) = far_end(i, speak);
		  end
		end

		% Shift memory
		st.X = circshift(st.X, [0, 0, 1]);

		for speak = 1:K
		%  Convert x (echo input) to frequency domain
		% MATLAB_MATCH: we divide by N to get values as in speex
		st.X(:, speak, 1) = fft(st.x(:, speak)) /N;
		end

		Sxx = 0;
		for speak = 1:K
		  Sxx = Sxx + sum(st.x(st.frame_size+1:end, speak).^2);
		  st.Xf = abs(st.X(1:st.frame_size+1, speak, 1)).^2;
		end

		Sff = 0;
		for chan = 1:C

		%  Compute foreground filter
		st.Y(:, chan) = 0;
		for speak=1:K
		  for j=1:M
		    st.Y(:, chan) = st.Y(:, chan) + st.X(:, speak, j) .* st.foreground(:, speak, j, chan);
		  end
		end
		% MATLAB_MATCH: we multiply by N to get values as in speex
		st.e(:, chan) = ifft(st.Y(:, chan)) * N;
		st.e(1:st.frame_size, chan) = st.input(:, chan) - st.e(st.frame_size+1:end, chan);
		% st.e : [out foreground | leak foreground ]
		Sff = Sff + sum(abs(st.e(1:st.frame_size, chan)).^2);

		end

		% Adjust proportional adaption rate */
    if (st.adapted)
		  st.prop = mdf_adjust_prop (st.W, N, M, C, K);
		end

		% Compute weight gradient */
if (st.saturated == 0)
		for chan = 1:C
		  for speak = 1:K
		    for j=M:-1:1
		      st.PHI = [st.power_1; st.power_1(end-1:-1:2)] .* st.prop(j) .* conj(st.X(:, speak, (j+1))) .* st.E(:, chan);
		      st.W(:, j) = st.W(:, j) + st.PHI;
		    end
		  end
		end
		else
		    st.saturated = st.saturated -1;
		end

		%FIXME: MC conversion required */
% Update weight to prevent circular convolution (MDF / AUMDF)
		for chan = 1:C
		for speak = 1:K
		for j = 1:M
		% This is a variant of the Alternatively Updated MDF (AUMDF) */
		% Remove the "if" to make this an MDF filter */
if (j==1 || mod(2+st.cancel_count,(M-1)) == j)
		st.wtmp = ifft(st.W(:, speak, j, chan));
		st.wtmp(st.frame_size+1:N) = 0;
		st.W(:, speak, j, chan) = fft(st.wtmp);
		end
		end
		end
		end

		% So we can use power_spectrum_accum */
		st.Yf = zeros(st.frame_size+1, 1);
		st.Rf = zeros(st.frame_size+1, 1);
		st.Xf = zeros(st.frame_size+1, 1);

		Dbf = 0;

		for chan = 1:C
		st.Y(:, chan) = 0;
		for speak=1:K
		for j=1:M
		st.Y(:, chan) = st.Y(:, chan) + st.X(:, speak, j) .* st.W(:, speak, j, chan);
		end
		end
		% MATLAB_MATCH: we multiply by N to get values as in speex
		st.y(:,chan) = ifft(st.Y(:,chan)) * N;
		% st.y : [ ~ | leak background ]
		end

		See = 0;

		% Difference in response, this is used to estimate the variance of our residual power estimate */
		for chan = 1:C
		st.e(1:st.frame_size, chan) = st.e(st.frame_size+1:N, chan) - st.y(st.frame_size+1:N, chan);
		Dbf = Dbf + 10 + sum(abs(st.e(1:st.frame_size, chan)).^2);
		st.e(1:st.frame_size, chan) = st.input(:, chan) - st.y(st.frame_size+1:N, chan);
		% st.e : [ out background | leak foreground ]
		See = See + sum(abs(st.e(1:st.frame_size, chan)).^2);
		end

		% Logic for updating the foreground filter */

		% For two time windows, compute the mean of the energy difference, as well as the variance */
		VAR1_UPDATE = .5;
		VAR2_UPDATE = .25;
		VAR_BACKTRACK = 4;
		MIN_LEAK = .005;

		st.Davg1 = .6*st.Davg1 + .4*(Sff-See);
		st.Davg2 = .85*st.Davg2 + .15*(Sff-See);
		st.Dvar1 = .36*st.Dvar1 + .16*Sff*Dbf;
		st.Dvar2 = .7225*st.Dvar2 + .0225*Sff*Dbf;

		update_foreground = 0;

		% Check if we have a statistically significant reduction in the residual echo */
		% Note that this is *not* Gaussian, so we need to be careful about the longer tail */
if (Sff-See)*abs(Sff-See) > (Sff*Dbf)
		update_foreground = 1;
elseif (st.Davg1* abs(st.Davg1) > (VAR1_UPDATE*st.Dvar1))
		update_foreground = 1;
elseif (st.Davg2* abs(st.Davg2) > (VAR2_UPDATE*(st.Dvar2)))
		update_foreground = 1;
		end

		% Do we update? */
if (update_foreground)

		st.Davg1 = 0;
		st.Davg2 = 0;
		st.Dvar1 = 0;
		st.Dvar2 = 0;
		st.foreground = st.W;
		% Apply a smooth transition so as to not introduce blocking artifacts */
		for chan = 1:C
		st.e(st.frame_size+1:N, chan) = (st.window(st.frame_size+1:N) .* st.e(st.frame_size+1:N, chan)) + (st.window(1:st.frame_size) .* st.y(st.frame_size+1:N, chan));
		end
		else
		reset_background=0;
		% Otherwise, check if the background filter is significantly worse */

if (-(Sff-See)*abs(Sff-See)> VAR_BACKTRACK*(Sff*Dbf))
		reset_background = 1;
		end
if ((-st.Davg1 * abs(st.Davg1))> (VAR_BACKTRACK*st.Dvar1))
		reset_background = 1;
		end
if ((-st.Davg2* abs(st.Davg2))> (VAR_BACKTRACK*st.Dvar2))
		reset_background = 1;
		end

if (reset_background)

		% Copy foreground filter to background filter */
		st.W = st.foreground;

		% We also need to copy the output so as to get correct adaptation */
		for chan = 1:C
		st.y(st.frame_size+1:N, chan) = st.e(st.frame_size+1:N, chan);
		st.e(1:st.frame_size, chan) = st.input(:, chan) - st.y(st.frame_size+1:N, chan);
		end

		See = Sff;
		st.Davg1 = 0;
		st.Davg2 = 0;
		st.Dvar1 = 0;
		st.Dvar2 = 0;
		end
		end

		Sey = 0;
		Syy = 0;
		Sdd = 0;

		for chan = 1:C

		% Compute error signal (for the output with de-emphasis) */
		for i=1:st.frame_size
		tmp_out = st.input(i, chan)- st.e(i+st.frame_size, chan);
		tmp_out = tmp_out + st.preemph * st.memE(chan);
		%  This is an arbitrary test for saturation in the microphone signal */
		if (in(i,chan) <= -32000 || in(i,chan) >= 32000)
if (st.saturated == 0)
		st.saturated = 1;
		end
		end
		out(i, chan) = tmp_out;
		st.memE(chan) = tmp_out;
		end

		% Compute error signal (filter update version) */
		st.e(st.frame_size+1:N, chan) = st.e(1:st.frame_size, chan);
		st.e(1:st.frame_size, chan) = 0;
		% st.e : [ zeros | out background ]

		% Compute a bunch of correlations */
		% FIXME: bad merge */
		Sey = Sey + sum(st.e(st.frame_size+1:N, chan) .* st.y(st.frame_size+1:N, chan));
		Syy = Syy + sum(st.y(st.frame_size+1:N, chan).^2);
		Sdd = Sdd + sum(st.input.^2);

		% Convert error to frequency domain */
		% MATLAB_MATCH: we divide by N to get values as in speex
		st.E = fft(st.e) / N;

		st.y(1:st.frame_size, chan) = 0;
		% MATLAB_MATCH: we divide by N to get values as in speex
		st.Y = fft(st.y) / N;

		% Compute power spectrum of echo (X), error (E) and filter response (Y) */
		st.Rf = abs(st.E(1:st.frame_size+1,chan)).^2;
		st.Yf = abs(st.Y(1:st.frame_size+1,chan)).^2;
		end

		% Do some sanity check */
if (~(Syy>=0 && Sxx>=0 && See >= 0))
		% Things have gone really bad */
		st.screwed_up = st.screwed_up + 50;
		out = out*0;
		elseif Sff > Sdd+ N*10000
		% AEC seems to add lots of echo instead of removing it, let's see if it will improve */
		st.screwed_up = st.screwed_up + 1;
		else
		% Everything's fine */
		st.screwed_up=0;
		end

if (st.screwed_up>=50)
		disp('Screwed up, full reset');
		st = speex_echo_state_reset_mdf(st);
		end

		% Add a small noise floor to make sure not to have problems when dividing */
		See = max(See, N* 100);

		for speak = 1:K
		Sxx = Sxx + sum(st.x(st.frame_size+1:end, speak).^2);
		st.Xf = abs(st.X(1:st.frame_size+1, speak, 1)).^2;
		end

		% Smooth far end energy estimate over time */
		st.power = ss_1*st.power+ 1 + ss*st.Xf;

		% Compute filtered spectra and (cross-)correlations */

		Eh_cur = st.Rf - st.Eh;
		Yh_cur = st.Yf - st.Yh;
		Pey_cur = Pey_cur + sum(Eh_cur.*Yh_cur) ;
		Pyy_cur = Pyy_cur + sum(Yh_cur.^2);
		st.Eh = (1-st.spec_average)*st.Eh + st.spec_average*st.Rf;
		st.Yh = (1-st.spec_average)*st.Yh + st.spec_average*st.Yf;

		Pyy = sqrt(Pyy_cur);
		Pey = Pey_cur/Pyy;

		% Compute correlation updatete rate */
		tmp32 = st.beta0*Syy;
if (tmp32 > st.beta_max*See)
		tmp32 = st.beta_max*See;
		end
		alpha = tmp32/ See;
		alpha_1 = 1- alpha;

		% Update correlations (recursive average) */
		st.Pey = alpha_1*st.Pey + alpha*Pey;
		st.Pyy = alpha_1*st.Pyy + alpha*Pyy;

		if st.Pyy<1
		st.Pyy =1;
		end

		% We don't really hope to get better than 33 dB (MIN_LEAK-3dB) attenuation anyway */
		if st.Pey< MIN_LEAK * st.Pyy
		st.Pey = MIN_LEAK * st.Pyy;
		end

if (st.Pey> st.Pyy)
		st.Pey = st.Pyy;
		end

		% leak_estimate is the linear regression result */
		st.leak_estimate = st.Pey/st.Pyy;

		% This looks like a stupid bug, but it's right (because we convert from Q14 to Q15) */
if (st.leak_estimate > 16383)
		st.leak_estimate = 32767;
		end

		% Compute Residual to Error Ratio */
		RER = (.0001*Sxx + 3.*st.leak_estimate*Syy) / See;
		% Check for y in e (lower bound on RER) */
if (RER < Sey*Sey/(1+See*Syy))
		RER = Sey*Sey/(1+See*Syy);
		end
if (RER > .5)
		RER = .5;
		end

		% We consider that the filter has had minimal adaptation if the following is true*/
if (~st.adapted && st.sum_adapt > M && st.leak_estimate*Syy > .03*Syy)
		st.adapted = 1;
		end

if (st.adapted)
		% Normal learning rate calculation once we're past the minimal adaptation phase */
		for i=1:st.frame_size+1

		% Compute frequency-domain adaptation mask */
		r = st.leak_estimate*st.Yf(i);
		e = st.Rf(i)+1;
if (r>.5*e)
		r = .5*e;
		end
		r = 0.7*r + 0.3*(RER*e);
		%st.power_1[i] = adapt_rate*r/(e*(1+st.power[i]));*/
		st.power_1(i) = (r/(e*st.power(i)+10));
		end
		else
		% Temporary adaption rate if filter is not yet adapted enough */
		adapt_rate=0;

if (Sxx > N* 1000)

		tmp32 = 0.25* Sxx;
if (tmp32 > .25*See)
		tmp32 = .25*See;
		end
		adapt_rate = tmp32/ See;
		end
		st.power_1 = adapt_rate./(st.power+10);


		% How much have we adapted so far? */
		st.sum_adapt = st.sum_adapt+adapt_rate;
		end

		% FIXME: MC conversion required */
		st.last_y(1:st.frame_size) = st.last_y(st.frame_size+1:N);
if (st.adapted)
		% If the filter is adapted, take the filtered echo */
		st.last_y(st.frame_size+1:N) = in-out;
		end

		end

function [out,mem] = filter_dc_notch16(in, radius, len, mem)
		out = zeros(size(in));
		den2 = radius*radius + .7*(1-radius)*(1-radius);
		for i=1:len
		vin = in(i);
		vout = mem(1) + vin;
		mem(1) = mem(2) + 2*(-vin + radius*vout);
		mem(2) = vin - (den2*vout);
		out(i) = radius*vout; 
		end

		end

function prop = mdf_adjust_prop(W, N, M, C, K)
		prop = zeros(M,1);
		for i=1:M
		tmp = 1;
		for chan=1:C
		for speak=1:K
		tmp = tmp + sum(abs(W(1:N/2+1, K, i, C)).^2);
		end
		end
		prop(i) = sqrt(tmp);
		end
		max_sum = max(prop, 1);
		prop = prop + .1*max_sum;
		prop_sum = 1+sum(prop);
		prop = .99*prop / prop_sum;
		end

		% Resets echo canceller state */
function st = speex_echo_state_reset_mdf(st)

		st.cancel_count=0;
		st.screwed_up = 0;
		N = st.window_size;
		M = st.M;
		C=st.C;
		K=st.K;

		st.e = zeros(N, C);
		st.x = zeros(N, K);
		st.input = zeros(st.frame_size, C);
		st.y = zeros(N, C);
		st.last_y = zeros(N, C);
		st.Yf = zeros(st.frame_size+1, 1);
		st.Rf = zeros(st.frame_size+1, 1);
		st.Xf = zeros(st.frame_size+1, 1);
		st.Yh = zeros(st.frame_size+1, 1);
		st.Eh = zeros(st.frame_size+1, 1);

		st.X = zeros(N, K, M+1);
		st.Y = zeros(N, C);
		st.E = zeros(N, C);
		st.W = zeros(N, K, M, C);
		st.foreground = zeros(N, K, M, C);
		st.PHI = zeros(N, 1);
		st.power = zeros(st.frame_size+1, 1);
		st.power_1 = ones((st.frame_size+1), 1);
		st.window = zeros(N, 1);
		st.prop = zeros(M, 1);
		st.wtmp = zeros(N, 1);

		st.memX = zeros(K, 1);
		st.memD = zeros(C, 1);
		st.memE = zeros(C, 1);

		st.saturated = 0;
		st.adapted = 0;
		st.sum_adapt = 0;
		st.Pey = 1;
		st.Pyy = 1;
		st.Davg1 = 0;
		st.Davg2 = 0;
		st.Dvar1 = 0;
		st.Dvar2 = 0;


		end

function was_closed = update_waitbar_check_wasclosed(h, n, end_point, Fs)
		was_closed = 0;

		% update waitbar
		try
		waitbar(n/end_point, h, ['Processing... ', num2str(n/Fs, '%.2f'), 's / ', num2str(end_point/Fs, '%.2f'), 's' ]);
		catch ME
		% if it's no longer there (closed by user)
		if (strcmp(ME.identifier(1:length('MATLAB:waitbar:')), 'MATLAB:waitbar:'))
		was_closed = 1; % then get out of the loop
		end
		end

		end

		end
