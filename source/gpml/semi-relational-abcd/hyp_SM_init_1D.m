function [hypout] = hyp_SM_init_1D(covtype,Q,x,y)
hypout = [];
switch(covtype)
    case 'covSMfast'
        
        % NOTE TO USER: SET FS= 1/[MIN GRID SPACING] FOR YOUR APPLICATION
        % Fs is the sampling rate
        Fs = 1;   % 1/[grid spacing].
        
        % Deterministic weights (fraction of variance)
        % Set so that k(0,0) is close to the empirical variance of the data.
        
        wm = std(y);
        w0 = wm/sqrt(Q)*ones(Q,1);
        
        w0 = w0.^2; % parametrization for covSMfast
        
        hypout = [w0];
        
        % Uniform random frequencies
        % Fs/2 will typically be the Nyquist frequency
        mu = max(Fs/2*rand(Q,1),1e-8);
        
        hypout = [hypout; mu];
        
        % Truncated Gaussian for length-scales (1/Sigma)
        sigmean = length(unique(x))*sqrt(2*pi)/2;
        hypout = [hypout; 1./(abs(sigmean*randn(Q,1)))];
end
end

