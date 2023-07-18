function X = rand_3D_sphere_points(n, density)
% Input parameters:
%   n       : Number of points
%   density : Point density on unit surface area, default is 100,
%             < 0 will generate points on a unit sphere
% Output parameter:
%   X : Size n * 3, each row is a point coordinate
if (nargin < 2), density = 100; end
X = rand(n, 3) - 0.5;
X = normr(X);
if (density > 0)
    r = sqrt(n / (4 * pi * density));
    X = X .* r;
end
end