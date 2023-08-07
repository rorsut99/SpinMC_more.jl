using Random
using LinearAlgebra

"""
returns a normalized vector of length dim complex numbers
"""
function uniformOnSphere(dim::Int64)
    vec=rand(Complex{Float64}, dim)
    return (vec/=LinearAlgebra.norm(vec))
end

"""
proposes update of spin state. There are two methods of proposing new spins: performing a rotation 
using a randomly chosen generator, or picking 2*dim random numbers.
Note: the rotation method is slower due to the exponential of a matrix, but converges faster
"""
function proposeUpdate(site,lattice::Lattice{D,N,dim,phdim},gens::Generators, rng = Random.GLOBAL_RNG) where {D,N,dim,phdim} 
    s1=getSpin(lattice, site)
    genIn=rand(1:gens.dim^2-1)
    gen=gens.generators[genIn]

    phi = 2.0 * pi * rand()
    rot=exp(1im*phi*gen)

    return (rot*s1)
end

# function proposeUpdate(site::Int64, lattice::Lattice{D,N,dim,phdim}, gens::Generators, rng = Random.GLOBAL_RNG) where {D,N,dim,phdim}
#     theta = rand(2)
#     phi = rand(3)*2*pi

#     x1 = theta[2]^(1/4)*theta[1]^(1/2)*sin(phi[1])
#     x2 = theta[2]^(1/4)*theta[1]^(1/2)*cos(phi[1])
#     x3 = theta[2]^(1/4)*sqrt(1-theta[1])*sin(phi[2])
#     x4 = theta[2]^(1/4)*sqrt(1-theta[1])*cos(phi[2])
#     x5 = sqrt(1-sqrt(theta[2]))*sin(phi[3])
#     x6 = sqrt(1-sqrt(theta[2]))*cos(phi[3])

#     d = [x1+x2*im, x3+x4*im, x5+x6*im]

#     # d = uniformOnSphere(gens.dim)
#     return d

#     # vec=rand(Complex{Float64}, gens.dim)
#     # return (vec/=LinearAlgebra.norm(vec))

# end

"""
calculates inner product of s1.gen.s2
"""
function calcInnerProd(s1::Vector{ComplexF64}, gen::Matrix{ComplexF64}, s2::Vector{ComplexF64})
    ret=dot(s1,gen*s2)
    # print(ret.im, "\n")
    # abs(ret.im) < 1e-6 || error(string("Imaginary part of inner product is too large."))
    return ret
end

"""
returns spin vector of length dim^2 of expectation values of generators for a spin s1
"""
function genExpVals(s1::Vector{ComplexF64}, gens::Generators) 
    vals=zeros(ComplexF64, gens.dim^2)
    i=1
    for i in 1:length(gens.generators)
        vals[i]=calcInnerProd(s1,gens.generators[i],s1)
    end
    return(vals)
end

"""
calculates the exchnage energy between s1 and s2
"""
function exchangeEnergy(s1::Vector{ComplexF64}, M::InteractionMatrix, s2::Vector{ComplexF64})
    ret = dot(s1, M.mat*s2)
    abs(ret.im) < 1e-6 || error(string("Imaginary part of inner product is too large."))
    return real(ret)
end

"""
calculates current energy of the entire system (spins + phonons)
"""
function getEnergy(lattice::Lattice{D,N,dim,phdim}, gens::Generators)::Float64 where {D,N,dim,phdim}
    energy = 0.0
    for site in 1:length(lattice)
        # get vector of exp values for site
        s0 = genExpVals(getSpin(lattice, site), gens)
        p0 = getPhonon(lattice, site)

        #two-spin interactions
        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        for i in 1:length(interactionSites)
            # get vector of exp values for interaction site
            s1 = genExpVals(getSpin(lattice, interactionSites[i]), gens)
            if site > interactionSites[i]
                energy += exchangeEnergy(s0, interactionMatrices[i], s1)
            end
        end

        energy += phononPotentialEnergy(lattice, p0)
        energy += spinPhononCoupling(lattice, s0, p0)

        #onsite interaction
       # energy += exchangeEnergy(s0, getInteractionOnsite(lattice, site), s0)

        #field interaction
        #energy += dot(s0, getInteractionField(lattice, site))
    end

    return energy
end

"""
returns energy difference between system with current spin and proposed new spin (newState)
"""
function getSpinEnergyDifference(lattice::Lattice{D,N,dim,phdim}, gens::Generators, site::Int, newState::Vector{ComplexF64})::Float64 where {D,N,dim,phdim}
    dE = 0.0
    E1 = 0.0    # new energy
    E2 = 0.0    # old energy

    # get spins and phonon
    oldState = getSpin(lattice, site)
    s1=genExpVals(newState,gens)
    s2=genExpVals(oldState,gens)
    p1 = getPhonon(lattice, site)

    #two-spin interactions
    interactionSites = getInteractionSites(lattice, site)
    interactionMatrices = getInteractionMatrices(lattice, site)
    for i in 1:length(interactionSites)
        s3 = genExpVals(getSpin(lattice, interactionSites[i]),gens)
        E1 += exchangeEnergy(s1, interactionMatrices[i], s3)
        E2 += exchangeEnergy(s2, interactionMatrices[i], s3)
    end

    E1 += spinPhononCoupling(lattice, s1, p1)
    E2 += spinPhononCoupling(lattice, s2, p1)
    #onsite interaction
    #interactionOnsite = getInteractionOnsite(lattice, site)
    #dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

    #field interaction
    #dE += dot(ds, getInteractionField(lattice, site))
    dE=E1-E2
    return dE
end

"""
returns energy difference between system with current phonon coordinate and proposed new phonon coordiante (newPhState)
"""
function getPhononEnergyDifference(lattice::Lattice{D,N,dim,phdim}, gens::Generators, site::Int, newPhState::Vector{Float64})::Float64 where {D,N,dim,phdim}
    dE = 0.0

    # get spin and phonon
    oldState = getSpin(lattice, site)
    s2=genExpVals(oldState,gens)
    p1 = getPhonon(lattice, site)

    dE += (phononPotentialEnergy(lattice, newPhState) - phononPotentialEnergy(lattice, p1))
    dE += (spinPhononCoupling(lattice, s2, newPhState) - spinPhononCoupling(lattice, s2, p1))

    #onsite interaction
    #interactionOnsite = getInteractionOnsite(lattice, site)
    #dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

    #field interaction
    #dE += dot(ds, getInteractionField(lattice, site))

    return dE
end

function getMagnetization(lattice::Lattice{D,N,dim,phdim},gens::Generators) where {D,N,dim,phdim}
    mag = zeros(gens.dim^2-1)
    for i in 1:length(lattice)
        spin = genExpVals(getSpin(lattice, i),gens)
        mag+=spin
    end
    return mag / length(lattice)
end

function getAFMMagnetization(lattice::Lattice{D,N,dim,phdim},gens::Generators) where {D,N,dim,phdim}
    mag = zeros(gens.dim^2-1)
    for i in 1:length(lattice)
        if i % 2 == 0
            spin = genExpVals(getSpin(lattice, i),gens)
        else
            spin = -1*genExpVals(getSpin(lattice, i),gens)
        end
        mag+=spin
    end
    return mag / length(lattice)
end

function getCorrelation(lattice::Lattice{D,N,dim,phdim},gens::Generators, spin::Int = 1) where {D,N,dim,phdim}
    corr = zeros(gens.dim^2, gens.dim^2, length(lattice))
    s = getSpin(lattice, spin)
    s0 = genExpVals(s, gens)

    for i in 1:length(lattice)
        state1 = getSpin(lattice, i)
        s1 = genExpVals(state1,gens)
        corr[:, :, i] = s0*transpose(s1)
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
    mag = length(lattice) .* getMagnetization(lattice,gens)
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

"""
calculates the spin vector of generator expactation values for the entire lattice and save the spins to lattice.expVals object
this function is to be used at the end of the MC to save the spin vectors for the time evolution.
"""
function finalState!(lattice::Lattice{D,N,dim,phdim},gens::Generators) where {D,N,dim,phdim}
    expVals=zeros(ComplexF64, gens.dim^2,length(lattice))
    for site in 1:length(lattice)
        s1=getSpin(lattice,site)
        vec=genExpVals(s1,gens)
        expVals[:,site]=vec
    end
    lattice.expVals=expVals
end

