from sympy import eye


def rank_at_most_one(matrix):
    pivot = next(((i, j) for i in range(matrix.rows)
                  for j in range(matrix.cols) if matrix[i, j] != 0), None) #find first nonzero entry
    if pivot is None:
        return True #zero matrix has rank 0
    r, c = pivot
    return all(matrix[i, j] * matrix[r, c] == matrix[i, c] * matrix[r, j]
               for i in range(matrix.rows) for j in range(matrix.cols)) #check condition for all entries


def classify_minimal(matrices):
    matrices = list(matrices.values())#turn dict to a list
    d = matrices[0].rows # d=n_min
    bound = d * (d - 1) // 2
    diagnostics = []
    if d == 1:
        return dict(order=0, minimal_dimension=d, search_bound=bound,
                    diagnostics=diagnostics)
    active = [eye(d)]
    for k in range(1, bound + 1):
        survivors = [] # prefixes with rank > 1
        for prefix in active:
            for matrix in matrices:
                product = prefix * matrix
                if not rank_at_most_one(product):
                    survivors.append(product)
        active = survivors # update active prefixes for next iteration
        diagnostics.append({"order": k, "surviving_products": len(active)})
        if not active: # return if active is empty meaning all products have rank at most 1
            return dict(order=k, minimal_dimension=d, search_bound=bound,
                        diagnostics=diagnostics)
    return dict(order=None, minimal_dimension=d, search_bound=bound,
                diagnostics=diagnostics) # return inf order if we reached the bound without finding a rank > 1 product
