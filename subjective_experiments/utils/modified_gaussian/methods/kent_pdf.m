function [ ret_val ] = kent_pdf( x, kappa, Beta, mean_center )
%KENT_PDF Compute pdf value of Kent distriubtion
%   For now we dont consider beta value
    %Note: if kappa is larger that 707 need high precision float pt
    if (kappa > 707)
        kappa = hpf(kappa);
    end
    norm_constant = (4*pi*sinh(kappa))/kappa;
    %check if matrix or vector
    [rows,cols] = size(x);
    if rows > 1
        %replicate mean center into matrix
        %then compute all dot products
        mean_center = repmat(mean_center,rows,1);
        dot_term = dot(x,mean_center,2);
    else
        dot_term = dot(x,mean_center);
    end
    ret_val = exp(kappa.*dot_term);
    ret_val = ret_val/norm_constant;
end

