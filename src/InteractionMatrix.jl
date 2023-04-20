#Changed definition of Interaction Matrix struct to hold dim by dim matrix of complex numbers

struct InteractionMatrix

    mat::Matrix{ComplexF64}
end

function InteractionMatrix(dim::Int)
    return InteractionMatrix(zeros(Float64,dim,dim)...)
end

# Checks inputted matrix is correct dimensions
function InteractionMatrix(M::T,dim::Int) where T<:AbstractMatrix
    size(M) == (dim,dim) || error(string("Interaction matrix must be of size ",dim,"x",dim,"."))

    return M
end