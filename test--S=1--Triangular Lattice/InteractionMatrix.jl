#Changed definition of Interaction Matrix struct to hold dim by dim matrix of complex numbers

struct InteractionMatrix

    mat::Matrix{Any}
end

function InteractionMatrix(dim::Int)
    return InteractionMatrix(zeros(Float64,dim^2-1,dim^2-1))
end

# Checks inputted matrix is correct dimensions
function InteractionMatrix(M::T,dim::Int) where T<:AbstractMatrix
    size(M) == (dim^2-1,dim^2-1) || error(string("Interaction matrix must be of size ",dim,"x",dim,"."))

    return M
end