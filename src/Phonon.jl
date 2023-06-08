using Random
using LinearAlgebra
using Distributions


function uniformDist(dim,Qmax)
    vec=rand(Uniform(-Qmax,Qmax), dim)
    return (vec)
end

function getPhonon(lattice::Lattice{D,N,dim,phdim}, site::Int) where {D,N,dim,phdim}
    return (lattice.phonons[:,site])
end

function phononPotentialEnergy(lattice::Lattice{D,N,dim,phdim},p1) where {D,N,dim,phdim}
    return (0.5*dot(lattice.springConstants,p1.^2))
end

function spinPhononCoupling(lattice::Lattice{D,N,dim,phdim},s1,p1) where {D,N,dim,phdim}
    return (dot(s1,lattice.phononCoupling*p1))
end

function setPhonon!(lattice::Lattice{D,N,dim,phdim}, site::Int, newState::Vector{Float64}) where {D,N,dim,phdim}
    lattice.phonons[:,site] = newState
end

function couplingMatrix(gens, M, dim, phdim)
    res=zeros(ComplexF64,dim^2,phdim)
    for i in 1:3
        for j in 1:phdim

            res[:,j]+=M[i,j]*gens.genReps[4,i]

        end
    end
    return res
end