function dz = dfx2D(x,y)
% gradient with respect to x of the function used for the 2D test 

% x MUST be [2,1]-sized
% y MUST be a column
    [yr, yc] = size(y);
    [xr, xc] = size(x);
    
    dz = zeros(2,yr);
    if (xr == 2)&&(xc == 1)&&(yc == 1)
        dz(1,:) = ( cos(x(1,:).*cos(x(1,:).*y)).*(cos(x(1,:).*y)-y.*x(1,:).*sin(x(1,:).*y))+...
                    x(2,:).*cos(x(1,:).*x(2,:)).*sin(x(2,:).*sin(y.*x(2,:))-...
                    sin(x(1,:).*x(2,:)))).*exp(0.5*(x(1,:)-2).^2+0.5*(x(2,:)-2).^2)+(x(1,:)-2).*f2D(x,y);
        
        dz(2,:) = -sin(x(2,:).*sin(x(2,:).*y)-sin(x(1,:).*x(2,:))).*(sin(x(2,:).*y)+...
            x(2,:).*y.*cos(x(2,:).*y)-x(1,:).*cos(x(1,:).*x(2,:))).*exp(0.5*(x(1,:)-2).^2+0.5*(x(2,:)-2).^2)+...
            (x(2,:)-2).*f2D(x,y);
    else
        error('wrong size of imput');
    end
end