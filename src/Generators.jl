using LinearAlgebra
mutable struct Generators
    generators::Vector{Matrix{ComplexF64}} #list of length dim^2-1 holding generators for the SU(N) representation of interest
    spinOperators::Vector{Matrix{ComplexF64}}
    genReps::Matrix{Vector{ComplexF64}}

    Generators() = new{}()
end

function initGen(dim)
    gens = Generators()
    gens.generators=[Matrix(1.0I,dim,dim)]
    gens.spinOperators=[Matrix(1.0I,dim,dim)]
    gens.genReps = Matrix{Vector{ComplexF64}}(undef,4,3)

    return gens
end

function addGenerator!(gens::Generators,M::Matrix{ComplexF64},d::Int64)
    size(M) == (d,d) || error(string("Generator must be of size ",d,"x",d,"."))

    if (length(gens.generators)==d^2-1)
        gens.generators[1]=M
    else
        push!(gens.generators,M)
    end
end

function addSpinOperator!(gens::Generators,M::Matrix{ComplexF64},d::Int64)
    if (length(gens.spinOperators)==3)
        gens.spinOperators[1]=M
    else
        push!(gens.spinOperators,M)
    end
end

function decomposeMat(gens::Generators,mat::Matrix{ComplexF64},d) 
    mats=copy(gens.generators)

    Id=Matrix((1.0+0im)I,d,d)
    push!(mats,Id)

    sols= vec(mat)
    eqs= Array{ComplexF64}(undef,d^2,d^2)

    for i in 1:d^2
        for j in 1:d^2
            eqs[i,j]=mats[j][i]
        end
    end

    return (eqs\sols)
end

function setGenReps!(gens::Generators,d)
    for i in 1:3
        for j in 1:3
            mat=gens.spinOperators[i]*gens.spinOperators[j]
            vec=decomposeMat(gens,mat,d)
            gens.genReps[i,j]=vec
        end
    end

    for k in 1:3
        mat=gens.spinOperators[k]
        vec=decomposeMat(gens,mat,d)
        gens.genReps[4,k]=vec
    end
end
