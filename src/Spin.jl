using Random
using LinearAlgebra

# Updated function returns a 'dim' dimensional vector of random complex numbers, divided by the norm
function uniformOnSphere(dim)
    vec=rand(Complex{Float64}, dim)
    return (vec/=LinearAlgebra.norm(vec))
end

# Created function to propose update of spin state
function proposeUpdate(site,lattice::Lattice{D,N,dim,phdim},gens::Generators,d, rng = Random.GLOBAL_RNG) where {D,N,dim,phdim} 
    s1=getSpin(lattice, site)
    genIn=rand(1:d^2-1)
    gen=gens.generators[genIn]


    phi = 2.0 * pi * rand(rng)
    rot=exp(1im*phi*gen)

    return (rot*s1)
end

# Created function to calculate inner product
function calcInnerProd(s1,gen,s2)
    ret=dot(s1,gen*s2)
    # if ret.im>1e-6
    #     print("Error")
    # end
    return real(ret)
end

#Created function to return vector of expctation values of all generators for a site
function genExpVals(s1,gens::Generators,d) 
    vals=zeros(d^2-1)
    i=1
    for mat in gens.generators
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
function getEnergy(lattice::Lattice{D,N,dim,phdim},gens::Generators)::Float64 where {D,N,dim,phdim}
    energy = 0.0
    d=size(gens.spinOperators[1])[1]
    Id=Matrix((1.0+0im)I,d,d)
    for site in 1:length(lattice)
        # get vector of exp values for site
        s0 = genExpVals(getSpin(lattice, site), gens,dim)
        p0 = getPhonon(lattice, site)
        tempS0=copy(s0)
        
        push!(tempS0,calcInnerProd(getSpin(lattice,site),Id,getSpin(lattice,site)))
    

        #two-spin interactions
        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        for i in 1:length(interactionSites)
            # get vector of exp values for interaction site
            s1 = genExpVals(getSpin(lattice, interactionSites[i]), gens,dim)
            tempS1=copy(s1)
            push!(tempS1,calcInnerProd(getSpin(lattice,interactionSites[i]),Id,getSpin(lattice,interactionSites[i])))
            if site > interactionSites[i]
                energy += exchangeEnergy(tempS0, interactionMatrices[i], tempS1)
            end
        end

        energy += phononPotentialEnergy(lattice, p0)
        energy += spinPhononCoupling(lattice, tempS0, p0)

        #onsite interaction
       # energy += exchangeEnergy(s0, getInteractionOnsite(lattice, site), s0)

        #field interaction
        #energy += dot(s0, getInteractionField(lattice, site))
    end

    return energy
end

#Updated to expect a vector of complex numbers for newState
function getSpinEnergyDifference(lattice::Lattice{D,N,dim,phdim},gens::Generators, site::Int, newState::Vector{ComplexF64})::Float64 where {D,N,dim,phdim}
    dE = 0.0
    oldState = getSpin(lattice, site)
    d=size(gens.spinOperators[1])[1]
    Id=Matrix((1.0+0im)I,d,d)


    s1=genExpVals(newState,gens,dim)
    s2=genExpVals(oldState,gens,dim)
    tempS1=copy(s1)
    tempS2=copy(s2)
    push!(tempS1,calcInnerProd(newState,Id,newState))
    push!(tempS2,calcInnerProd(oldState,Id,oldState))
    
    ds = s1 .- s2

    p1 = getPhonon(lattice, site)

    #two-spin interactions
    interactionSites = getInteractionSites(lattice, site)
    interactionMatrices = getInteractionMatrices(lattice, site)
    E1=0
    E2=0
    for i in 1:length(interactionSites)
        s3 = genExpVals(getSpin(lattice, interactionSites[i]),gens,dim)
        tempS3=copy(s3)
        push!(tempS3,calcInnerProd(getSpin(lattice, interactionSites[i]),Id,getSpin(lattice, interactionSites[i])))



        E1 += exchangeEnergy(tempS1, interactionMatrices[i], tempS3)

        E2 += exchangeEnergy(tempS2, interactionMatrices[i], tempS3)
    end

    E1 += spinPhononCoupling(lattice, tempS1, p1)
    E2 += spinPhononCoupling(lattice, tempS2, p1)
    #onsite interaction
    #interactionOnsite = getInteractionOnsite(lattice, site)
    #dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

    #field interaction
    #dE += dot(ds, getInteractionField(lattice, site))
    dE=E1-E2
    return dE
end

function getPhononEnergyDifference(lattice::Lattice{D,N,dim,phdim}, gens::Generators, site::Int, newPhState::Vector{Float64})::Float64 where {D,N,dim,phdim}
    dE = 0.0
    oldState = getSpin(lattice, site)

    Id=Matrix((1.0+0im)I,dim,dim)
    s2=genExpVals(oldState,gens,dim)
    tempS2=copy(s2)
    push!(tempS2,calcInnerProd(oldState,Id,oldState))

    p1 = getPhonon(lattice, site)

    dE += (phononPotentialEnergy(lattice, newPhState) - phononPotentialEnergy(lattice, p1))
    dE += (spinPhononCoupling(lattice, tempS2, newPhState) - spinPhononCoupling(lattice, tempS2, p1))

    #onsite interaction
    #interactionOnsite = getInteractionOnsite(lattice, site)
    #dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

    #field interaction
    #dE += dot(ds, getInteractionField(lattice, site))

    return dE
end

function getMagnetization(lattice::Lattice{D,N,dim,phdim},gens::Generators,d) where {D,N,dim,phdim}
    mag = zeros(d^2-1)
    for i in 1:length(lattice)
        spin = genExpVals(getSpin(lattice, i),gens,d)
        mag+=spin
    end
    return mag / length(lattice)
end

function getAFMMagnetization(lattice::Lattice{D,N,dim,phdim},gens::Generators,d) where {D,N,dim,phdim}
    mag = zeros(d^2-1)
    for i in 1:length(lattice)
        if i % 2 == 0
            spin = genExpVals(getSpin(lattice, i),gens,d)
        else
            spin = -1*genExpVals(getSpin(lattice, i),gens,d)
        end
        mag+=spin
    end
    return mag / length(lattice)
end

function getCorrelation(lattice::Lattice{D,N,dim,phdim},gens::Generators,d, spin::Int = 1) where {D,N,dim,phdim}
    corr = zeros(d^2, d^2, length(lattice))
    Id=Matrix((1.0+0im)I,d,d)
    s = getSpin(lattice, spin)
    s0 = genExpVals(s, gens, d)
    tempS0=copy(s0)
    push!(tempS0,calcInnerProd(s,Id,s))
    for i in 1:length(lattice)
        state1 = getSpin(lattice, i)
        s1 = genExpVals(state1,gens,d)
        tempS1=copy(s1)
        push!(tempS1,calcInnerProd(state1,Id,state1))
        corr[:, :, i] = tempS0*transpose(tempS1)
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

function getSusceptibility(lattice::Lattice{D,N,dim,phdim},d) where {D,N,dim,phdim}
    chitens = zeros(Float64,3,3)
    mag = length(lattice) .* getMagnetization(lattice,gens,d)
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


function finalState!(lattice::Lattice{D,N,dim,phdim},gens::Generators,d) where {D,N,dim,phdim}
    expVals=zeros(dim^2,length(lattice))
    Id=Matrix(1.0I,d,d)
    for site in 1:length(lattice)
        s1=getSpin(lattice,site)
        vec=genExpVals(s1,gens,d)
        push!(vec,calcInnerProd(getSpin(lattice,site),Id,getSpin(lattice,site)))
        expVals[:,site]=vec
    end
    lattice.expVals=expVals
end



# Compute interaction of two sites where inter1, inter2 are integers representing direction 1=z, 2=x,3=y
function genRepInteraction(lattice::Lattice{D,N,dim,phdim},gens::Generators, inter1, inter2, s0, s1,site1,site2,d) where {D,N,dim,phdim}
    J=-1.0
    Id=Matrix(1.0I,d,d)
    tempS0=copy(s0)
    tempS1=copy(s1)
    push!(tempS0,calcInnerProd(getSpin(lattice,site1),Id,getSpin(lattice,site1)))
    push!(tempS1,calcInnerProd(getSpin(lattice,site2),Id,getSpin(lattice,site2)))


    spin1=dot(gens.genReps[4,inter1],tempS0)
    spin2=dot(gens.genReps[4,inter2],tempS1)
    res=spin1*spin2

    if(res.im>1e-6)
        print("Error\n")
    end

    return (J*res.re)
end



# Compute (S1⋅S2)^2 term
function quadSpinInteraction(lattice::Lattice{D,N,dim,phdim},gens::Generators, s0, s1,site1,site2,d) where {D,N,dim,phdim}
    K=0.0
    Id=Matrix(1.0I,d,d)
    tempS0=copy(s0)
    tempS1=copy(s1)
    push!(tempS0,calcInnerProd(getSpin(lattice,site1),Id,getSpin(lattice,site1)))
    push!(tempS1,calcInnerProd(getSpin(lattice,site2),Id,getSpin(lattice,site2)))
    


    res=0.0
    for i in 1:3
        for j in 1:3
            spin1=dot(gens.genReps[i,j],tempS0)
            spin2=dot(gens.genReps[i,j],tempS1)
            res+=spin1*spin2
            end
    end

  




    # if(res.im>1e-6)
    #     print("Error\n")
    # end

    return (-K*res.re)
end







