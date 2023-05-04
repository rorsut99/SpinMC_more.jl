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

function phononPotentialEnergy(lattice::Lattice{D,N,dim,phdim},site) where {D,N,dim,phdim}
    p1=getPhonon(lattice,site)
    return (0.5*dot(lattice.springConstants,p1.^2))
end

function spinPhononCoupling(lattice::Lattice{D,N,dim,phdim},s1,site) where {D,N,dim,phdim}
    p1=getPhonon(lattice,site)
    return (dot(s1,lattice.phononCoupling*p1))
end