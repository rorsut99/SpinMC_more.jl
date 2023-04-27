using Random
using LinearAlgebra

# Updated function returns a 'dim' dimensional vector of random complex numbers, divided by the norm
function uniformOnSphere(dim,rng = Random.GLOBAL_RNG)
    vec=rand(Complex{Float64}, dim)
    return (vec/=LinearAlgebra.norm(a))
end

# Created function to propose update of spin state
function proposeUpdate(site,dim,lattice::Lattice{D,N,dim}, rng = Random.GLOBAL_RNG)
    s1=getSpin(lattice, site)
    genIn=rand(1:dim)
    gen=lattice.generators[genIn]

    phi = 2.0 * pi * rand(rng)
    rot=exp(1im*phi*gen)

    return (rot*s1)
end

# Created function to calculate inner product
function calcInnerProd(s1,gen,s2)
    res=gen*s2
    s3=conj(s1)
    return(dot(s3,res))
end

#Created function to return vector of expctation values of all generators for a site
function genExpVals(s1,lattice::Lattice{D,N,dim},dim)
    vals=zeros(dim)
    i=0
    for mat in lattice.generators
        vals[i]=calcInnerProd(s1,mat,s1)
        i+=1
    end
    return(vals)
end

# this is shorter but redundant
function exchangeEnergy(s1, M::InteractionMatrix, s2)::Float64
    return calcInnerProd(s1, M, s2)
end

# calculates energy in terms of exp values vectors
function getEnergy(lattice::Lattice{D,N,dim})::Float64 where {D,N}
    energy = 0.0

    for site in 1:length(lattice)
        # get vector of exp values for site
        s0 = genExpVals(getSpin(lattice, site), lattice, dim)

        #two-spin interactions
        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        for i in 1:length(interactionSites)
            # get vector of exp values for interaction site
            s1 = genExpVals(getSpin(lattice, interactionSites[i]), lattice, dim)
            if site > interactionSites[i]
                energy += exchangeEnergy(s0, interactionMatrices[i], s1)
            end
        end

        #onsite interaction
        energy += exchangeEnergy(s0, getInteractionOnsite(lattice, site), s0)

        #field interaction
        energy += dot(s0, getInteractionField(lattice, site))
    end

    return energy
end

#Updated to expect a vector of complex numbers for newState
function getEnergyDifference(lattice::Lattice{D,N,dim}, site::Int, newState::Vector{ComplexF64})::Float64 where {D,N}
    dE = 0.0
    oldState = getSpin(lattice, site)
    ds = newState .- oldState

    #two-spin interactions
    interactionSites = getInteractionSites(lattice, site)
    interactionMatrices = getInteractionMatrices(lattice, site)
    for i in 1:length(interactionSites)
        dE += exchangeEnergy(ds, interactionMatrices[i], getSpin(lattice, interactionSites[i]))
    end

    #onsite interaction
    interactionOnsite = getInteractionOnsite(lattice, site)
    dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

    #field interaction
    dE += dot(ds, getInteractionField(lattice, site))

    return dE
end

function getMagnetization(lattice::Lattice{D,N,dim}) where {D,N}
    mx, my, mz = 0.0, 0.0, 0.0
    for i in 1:length(lattice)
        spin = getSpin(lattice, i)
        mx += spin[1]
        my += spin[2]
        mz += spin[3]
    end
    return [mx, my, mz] / length(lattice)
end

function getCorrelation(lattice::Lattice{D,N,dim}, spin::Int = 1) where {D,N}
    corr = zeros(length(lattice))
    s0 = getSpin(lattice, spin)
    for i in 1:length(lattice)
        corr[i] = dot(s0, getSpin(lattice, i))
    end
    return corr
end


# function getSusceptibility(a::Int , b::Int , lattice::Lattice{D,N,dim}) where {D,N}
#     ans = 0.0
#     for j in 1:length(lattice)
#         s0 = getSpin(lattice, j)[a]
#         ans += sum( [s0*getSpin(lattice,i)[b] for i in 1:length(lattice) ] )
#     end
#     return ans
# end

function getSusceptibility(lattice::Lattice{D,N,dim}) where {D,N}
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


function finalState!(lattice::Lattice{D,N,dim},L::NTuple{D,Int}) where {D,N,dim}
    expVals::Vector{N,Vector{dim^2-1,Float64}}
    for site in 1:length(lattice)
        s1=getSpin(lattice,site)
        vec=genExpVals(s1,lattice,dim)
        expVals[site]=vec
    end
    push!(lattice.expVals,expVals)

end
                
