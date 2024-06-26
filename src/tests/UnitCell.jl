struct UnitCell{D}
    primitive::NTuple{D,NTuple{D,Float64}}
    basis::Vector{NTuple{D,Float64}}
    interactions::Vector{Tuple{Int,Int,NTuple{D,Int},Matrix{Float64}}} #interactions specified as (basis1,basis2,offsetPrimitive,M)
    interactionsOnsite::Vector{Matrix{Float64}}
    interactionsField::Vector{Vector{Float64}}

    UnitCell(a1::NTuple{1,Float64}) = new{1}((a1,), Vector{NTuple{1,Float64}}(undef,0), Vector{Tuple{Int,Int,NTuple{1,Int},Matrix{Float64}}}(undef,0), Vector{Matrix{Float64}}(undef,0), Vector{Vector{Float64}}(undef,0))
    UnitCell(a1::NTuple{2,Float64}, a2::NTuple{2,Float64}) = new{2}((a1,a2), Vector{NTuple{2,Float64}}(undef,0), Vector{Tuple{Int,Int,NTuple{2,Int},Matrix{Float64}}}(undef,0), Vector{Matrix{Float64}}(undef,0), Vector{Vector{Float64}}(undef,0))
    UnitCell(a1::NTuple{3,Float64}, a2::NTuple{3,Float64}, a3::NTuple{3,Float64}) = new{3}((a1,a2,a3), Vector{NTuple{3,Float64}}(undef,0), Vector{Tuple{Int,Int,NTuple{3,Int},Matrix{Float64}}}(undef,0), Vector{Matrix{Float64}}(undef,0), Vector{Vector{Float64}}(undef,0))
    UnitCell(primitives...) = new{length(primitives)}(primitives, Vector{NTuple{length(primitives),Float64}}(undef,0), Vector{Tuple{Int,Int,NTuple{length(primitives),Int},Matrix{Float64}}}(undef,0), Vector{Matrix{Float64}}(undef,0), Vector{Vector{Float64}}(undef,0))
end

"""
converts matrix M from spin basis to generator basis. The matrix defines the interaction between spin components and the order can be specified as 1 or 2.
Example: a Heisenberg ferromagnet would have M=Id and order=1. 
Example: a biquadratic interaction would have M=Id and order=2.
"""
function getInteractionMatrix(M,gens,order)
    if order==1
        res=zeros(ComplexF64,gens.dim^2,gens.dim^2)
        for i in 1:3
            for j in 1:3

                res+=M[i,j]*gens.genReps[4,i]*transpose(gens.genReps[4,j])

            end
        end

    end
    if order==2
        reps=transpose(gens.genReps)
        res=zeros(ComplexF64,gens.dim^2,gens.dim^2)
        newM=zeros(ComplexF64,3,3)
        for i in 1:3
            for j in 1:3
                if M[i,j] != 0
                    newM[i,j] = 1.0+0.0im
                end
            end
        end
        mat=kron(newM,M)
        for i in 1:9
            for j in 1:9
                res+=mat[i,j]*transpose(reps[i])*reps[j]
            end
        end


    end
    return (res)
end

"""
Adds an interaction between spin1 located at basis site `b1` of the given `unitcell` and spin2 at basis site `b2` in a unit cell that is offset by `offset` lattice vectors. 
The exchange energy is calculated as spin1'.M.spin2. 
"""
function addInteraction!(unitcell::UnitCell{D}, gens::Generators, b1::Int, b2::Int, M::Matrix{Float64}, order::Int, offset::NTuple{D,Int}=Tuple(zeros(Int,D))) where D
    size(M) == (3,3) || error(string("Interaction matrix must be of size 3x3."))
    b1 == b2 && offset == Tuple(zeros(Int,D)) && error("Interaction cannot be local. Use setInteractionOnsite!() instead.")

    # get interaction matrix in generator basis from matrix in spin basis
    interaction=getInteractionMatrix(M,gens,order)

    push!(unitcell.interactions, (b1,b2,offset,interaction))
end

"""
adds an on site interaction to basis site b
"""
function setInteractionOnsite!(unitcell::UnitCell{D}, b::Int, M::Matrix{Float64}, dim::Int) where D
    size(M) == (3,3) || error(string("Interaction matrix must be of size 3x3."))
    unitcell.interactionsOnsite[b] = M
end

"""
adds a field B to basis site b. The field must be input in spin basis.
"""
function setField!(unitcell::UnitCell{D},gens, b::Int, B::Vector{Float64}) where D
    size(B) == (3,) || error(string("Field must be a vector of length 3."))

    # get field in generator basis from spin basis
    vec=getField(B,gens)
    unitcell.interactionsField[b] = vec
end

"""
converts the field V to generator basis from spin basis
"""
function getField(V::Vector{Float64}, gens::Generators)
    res=zeros(gens.dim^2)
    for i in 1:length(V)
        res+=V[i]*gens.genReps[4,i]
    end

    return (res)
end

"""
adds a basis site in the unit cell at position with empty on site and field interactions
"""
function addBasisSite!(unitcell::UnitCell{D}, position::NTuple{D,Float64}, dim::Int) where D
    push!(unitcell.basis, position)
    push!(unitcell.interactionsOnsite, zeros(dim^2,dim^2))
    push!(unitcell.interactionsField, zeros(dim^2))
    return length(unitcell.basis)
end