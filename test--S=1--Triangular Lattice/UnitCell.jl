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
Adds an interaction between spin1 located at basis site `b1` of the given `unitcell` and spin2 at basis site `b2` in a unit cell that is offset by `offset` lattice vectors. 
The exchange energy is calculated as spin1'.M.spin2. 
"""



#Changed hard coded '3' to dim--dimension of spin vector
function addInteraction!(unitcell::UnitCell{D}, b1::Int, b2::Int, M::Matrix{Float64},order::Int, dim::Int, offset::NTuple{D,Int}=Tuple(zeros(Int,D))) where D
    size(M) == (dim^2-1,dim^2-1) || error(string("Interaction matrix must be of size ",dim^2-1,"x",dim^2-1,"."))
    b1 == b2 && offset == Tuple(zeros(Int,D)) && error("Interaction cannot be local. Use setInteractionOnsite!() instead.")

    # get interaction matrix in generator basis from matrix in spin basis

    push!(unitcell.interactions, (b1,b2,offset,M))
end

function setInteractionOnsite!(unitcell::UnitCell{D}, b::Int, M::Matrix{Float64},dim::Int) where D
    size(M) == (dim^2-1,dim^2-1) || error(string("Interaction matrix must be of size",dim,"x",dim,"."))
    unitcell.interactionsOnsite[b] = M
end

function setField!(unitcell::UnitCell{D}, b::Int, B::Vector{Float64},dim::Int) where D
    size(B) == (dim,) || error(string("Field must be a vector of length ",dim,"."))
    unitcell.interactionsField[b] = B
end

function addBasisSite!(unitcell::UnitCell{D}, position::NTuple{D,Float64},dim::Int) where D
    push!(unitcell.basis, position)
    push!(unitcell.interactionsOnsite, zeros(dim,dim))
    push!(unitcell.interactionsField, zeros(dim))
    return length(unitcell.basis)
end