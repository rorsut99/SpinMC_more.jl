
using LinearAlgebra

"""
Generator struct holds all objects associated with the SU(N) representation needed to perform MD.
"""
mutable struct Generators
    dim::Int64
    generators::Vector{Matrix{ComplexF64}}      # list of length dim^2 holding generators for the SU(N) representation of interest
    spinOperators::Vector{Matrix{ComplexF64}}   # holds dimxdim operators representing Sx, Sy and Sz
    genReps::Matrix{Vector{ComplexF64}}         # holds generator representation of spin operators

    Generators() = new{}()
end

"""
initializes empty generator object
"""
function initGen(dim::Int64)
    gens = Generators()
    gens.dim = dim
    gens.generators=Vector{Matrix{ComplexF64}}()
    gens.spinOperators=Vector{Matrix{ComplexF64}}()
    gens.genReps = Matrix{Vector{ComplexF64}}(undef,4,3)    # first 3x3 is second order, last col is first order

    return gens
end

"""
adds matrix M to gens.generators
"""
function addGenerator!(gens::Generators, M::Matrix{ComplexF64})
    size(M) == (gens.dim,gens.dim) || error(string("Generator must be of size ",gens.dim,"x",gens.dim,"."))
    length(gens.generators) != gens.dim^2 || error(string("Only ", gens.dim^2, " generators can be stored. Replace one using gens.generators[index]"))
    push!(gens.generators,M)
end

"""
adds matrix M to gens.spinOperators
"""
function addSpinOperator!(gens::Generators ,M::Matrix{ComplexF64})
    size(M) == (gens.dim,gens.dim) || error(string("Spin operator must be of size ",gens.dim,"x",gens.dim,"."))
    size(gens.spinOperators) != (3,) || error(string("Only 3 spin operators can be stored. Replace one using gens.spinOperators[index]."))
    push!(gens.spinOperators,M)
end

"""
decomposes the matrix mat into a linear combination of generators
returns a vector of dim^2 coefficients
"""
function decomposeMat(gens::Generators, mat::Matrix{ComplexF64}) 
    sols = vec(mat)     # matrix that is to be decomposed

    # creates array of generators to solve for coefficients
    eqs = Array{ComplexF64}(undef,gens.dim^2,gens.dim^2)
    for i in 1:gens.dim^2
        for j in 1:gens.dim^2
            eqs[i,j]=gens.generators[j][i]
        end
    end

    return (eqs\sols)   # solves coeffs*eqs=sols
end

"""
saves generator representation of spin operators to struct
3 is hard coded because there are 3 spin operators
"""
function setGenReps!(gens::Generators)
    # generator reps of second order in spin operators (ie. SxSx etc)
    for i in 1:3
        for j in 1:3
            mat=gens.spinOperators[i]*gens.spinOperators[j]
            vec=decomposeMat(gens,mat)
            gens.genReps[i,j]=vec
        end
    end

    # generator reps of first order in spin operators (Sx, Sy, Sz)
    for k in 1:3
        mat=gens.spinOperators[k]
        vec=decomposeMat(gens,mat)
        gens.genReps[4,k]=vec
    end
end
