using Random
using LinearAlgebra

# Updated function returns a 'dim' dimensional vector of random complex numbers, divided by the norm
function uniformOnSphere(dim)
    vec=rand(Complex{Float64}, dim)
    return (vec/=LinearAlgebra.norm(vec))
end

# Created function to propose update of spin state
function proposeUpdate(site,lattice::Lattice{D,N,dim,phdim},d, rng = Random.GLOBAL_RNG) where {D,N,dim,phdim} 
    s1=getSpin(lattice, site)
    genIn=rand(1:d^2-1)
    gen=lattice.generators[genIn]


    phi = 2.0 * pi * rand(rng)
    rot=exp(1im*phi*gen)

    return (rot*s1)
end

# Created function to calculate inner product
function calcInnerProd(s1,gen,s2)
    return real(dot(s1,gen*s2))
end

#Created function to return vector of expctation values of all generators for a site
function genExpVals(s1,lattice::Lattice{D,N,dim,phdim},d) where {D,N,dim,phdim}
    vals=zeros(d^2-1)
    i=1
    for mat in lattice.generators
        vals[i]=calcInnerProd(s1,mat,s1)
        i+=1
    end
    return(vals)
end

# this is shorter but redundant
function exchangeEnergy(s1, M::InteractionMatrix, s2)::Float64
    return calcInnerProd(s1, M.mat, s2)
end

# calculates energy in terms of exp values vectors
function getEnergy(lattice::Lattice{D,N,dim,phdim})::Float64 where {D,N,dim,phdim}
    energy = 0.0

    for site in 1:length(lattice)
        # get vector of exp values for site
        s0 = genExpVals(getSpin(lattice, site), lattice,dim)
        # p0 = getPhonon(lattice, site)

        #two-spin interactions
        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        for i in 1:length(interactionSites)
            # get vector of exp values for interaction site
            s1 = genExpVals(getSpin(lattice, interactionSites[i]), lattice,dim)
            if site > interactionSites[i]
                energy += exchangeEnergy(s0, interactionMatrices[i], s1)
            end
        end

        # energy += phononPotentialEnergy(lattice, p0)
        # energy += spinPhononCoupling(lattice, s0, p0)

        #onsite interaction
       # energy += exchangeEnergy(s0, getInteractionOnsite(lattice, site), s0)

        #field interaction
        #energy += dot(s0, getInteractionField(lattice, site))
    end

    return energy
end

#Updated to expect a vector of complex numbers for newState
function getSpinEnergyDifference(lattice::Lattice{D,N,dim,phdim}, site::Int, newState::Vector{ComplexF64})::Float64 where {D,N,dim,phdim}
    dE = 0.0
    oldState = getSpin(lattice, site)

    s1=genExpVals(newState,lattice,dim)
    s2=genExpVals(oldState,lattice,dim)
    ds = s1 .- s2

    # p1 = getPhonon(lattice, site)

    #two-spin interactions
    interactionSites = getInteractionSites(lattice, site)
    interactionMatrices = getInteractionMatrices(lattice, site)
    for i in 1:length(interactionSites)
        dE += exchangeEnergy(ds, interactionMatrices[i],  genExpVals(getSpin(lattice, interactionSites[i]), lattice,dim))
    end

    # dE += (spinPhononCoupling(lattice, s1, p1) - spinPhononCoupling(lattice, s2, p1))

    #onsite interaction
    #interactionOnsite = getInteractionOnsite(lattice, site)
    #dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

    #field interaction
    #dE += dot(ds, getInteractionField(lattice, site))

    return dE
end

function getPhononEnergyDifference(lattice::Lattice{D,N,dim,phdim}, site::Int, newPhState::Vector{Float64})::Float64 where {D,N,dim,phdim}
    dE = 0.0
    oldState = getSpin(lattice, site)

    s2=genExpVals(oldState,lattice,dim)

    # p1 = getPhonon(lattice, site)

    # dE += (phononPotentialEnergy(lattice, newPhState) - phononPotentialEnergy(lattice, p1))
    # dE += (spinPhononCoupling(lattice, s2, newPhState) - spinPhononCoupling(lattice, s2, p1))

    #onsite interaction
    #interactionOnsite = getInteractionOnsite(lattice, site)
    #dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

    #field interaction
    #dE += dot(ds, getInteractionField(lattice, site))

    return dE
end

function getMagnetization(lattice::Lattice{D,N,dim,phdim},d) where {D,N,dim,phdim}
    mag = zeros(d^2-1)
    for i in 1:length(lattice)
        spin = genExpVals(getSpin(lattice, i),lattice,d)
        mag+=spin
    end
    return mag / length(lattice)
end

function getAFMMagnetization(lattice::Lattice{D,N,dim,phdim},d) where {D,N,dim,phdim}
    mag = zeros(d^2-1)
    for i in 1:length(lattice)
        if i % 2 == 0
            spin = genExpVals(getSpin(lattice, i),lattice,d)
        else
            spin = -1*genExpVals(getSpin(lattice, i),lattice,d)
        end
        mag+=spin
    end
    return mag / length(lattice)
end

function getCorrelation(lattice::Lattice{D,N,dim,phdim}, spin::Int = 1) where {D,N,dim,phdim}
    corr = zeros(length(lattice))
    s0 = getSpin(lattice, spin)
    for i in 1:length(lattice)
        corr[i] = dot(s0, getSpin(lattice, i))
    end
    return corr
end


# function getSusceptibility(a::Int , b::Int , lattice::Lattice{D,N,dim,phdim}) where {D,N,dim}
#     ans = 0.0
#     for j in 1:length(lattice)
#         s0 = getSpin(lattice, j)[a]
#         ans += sum( [s0*getSpin(lattice,i)[b] for i in 1:length(lattice) ] )
#     end
#     return ans
# end

function getSusceptibility(lattice::Lattice{D,N,dim,phdim}) where {D,N,dim,phdim}
    chitens = zeros(Float64,3,3)
    mag = length(lattice) .* getMagnetization(lattice)
    for k in 1:3
        for l in 1:3
            chitens[k,l] = mag[k]*mag[l]
        end
    end 
    # indices = [1,3]
    # Sp = lattice.spins[indices,:]
    # for j in 1:length(lattice)
    #         s0 = getSpin(lattice, j)
    #         for k in 1:2
    #             chitens[:,k] += sum(s0[indices[k]] .* Sp , dims=2)
    #         end
    # end
    return chitens
end


function finalState!(lattice::Lattice{D,N,dim,phdim},d) where {D,N,dim,phdim}
    expVals=Vector{Vector{Float64}}(undef,length(lattice))
    for site in 1:length(lattice)
        s1=getSpin(lattice,site)
        vec=genExpVals(s1,lattice,d)
        expVals[site]=vec
    end
    lattice.expVals=expVals

end


function decomposeMat(lattice::Lattice{D,N,dim,phdim},mat::Matrix{ComplexF64},d) where {D,N,dim,phdim}
    mats=copy(lattice.generators)

    push!(mats,(Matrix((1.0+0.0im)I,dim,dim)))

    sols= vec(mat)
    eqs= Array{ComplexF64}(undef,d^2,d^2)


    for i in 1:d^2
        for j in 1:d^2
            eqs[i,j]=mats[j][i]
        end
    end

    return (eqs\sols)
end

# Compute interaction of two sites where inter1, inter2 are integers representing direction 1=z, 2=x,3=y
function genRepInteraction(lattice::Lattice{D,N,dim,phdim}, inter1, inter2, site1::Int, site2::Int,d) where {D,N,dim,phdim}
    Id=Matrix(1.0I,d,d)
    
    s0=genExpVals(getSpin(lattice,site1),lattice,d)
    push!(s0,calcInnerProd(getSpin(lattice,site1),Id,getSpin(lattice,site1)))
    s1=genExpVals(getSpin(lattice,site2),lattice,d)
    push!(s1,calcInnerProd(getSpin(lattice,site1),Id,getSpin(lattice,site1)))

    spin1=dot(lattice.genReps[4,inter1],s0)
    spin2=dot(lattice.genReps[4,inter2],s1)
    return (spin1*spin2)
end

# Compute (S1⋅S2)^2 term
function quadSpinInteraction(lattice::Lattice{D,N,dim,phdim}, site1::Int, site2::Int,d) where {D,N,dim,phdim}
    Id=Matrix(1.0I,d,d)
    s0=genExpVals(getSpin(lattice,site1),lattice,d)
    push!(s0,calcInnerProd(getSpin(lattice,site1),Id,getSpin(lattice,site1)))
    s1=genExpVals(getSpin(lattice,site2),lattice,d)
    push!(s1,calcInnerProd(getSpin(lattice,site1),Id,getSpin(lattice,site1)))
    res=0.0
    for i in 1:9
        spin1=dot(lattice.genReps[i],s0)
        spin2=dot(lattice.genReps[i],s1)
        res+=spin1*spin2
    end

    return (res)
end








