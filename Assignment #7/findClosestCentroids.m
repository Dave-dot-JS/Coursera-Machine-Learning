function idx = findClosestCentroids(X, centroids)

K = size(centroids, 1);

idx = zeros(size(X,1), 1);

distance = zeros(K,1);

for i = 1:length(X) 
  for j = 1:K
    D = bsxfun(@minus, X(i,:), centroids(j,:));
    distance(j) = sum(D.^2,2);
  end
  [value, idx(i)] = min(distance);
end

end