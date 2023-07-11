using BinningAnalysis

mutable struct EvolveObservables


    spinStates::FullBinner{Matrix{Float64}}
    phononPosition::FullBinner{Matrix{Float64}}
    phononMomenta::FullBinner{Matrix{Float64}}
    energySeries::Vector{Float64}

    EvolveObservables()=new{}()
end

function initEvolveObservables()
    evsObs=EvolveObservables()

    evsObs.spinStates=FullBinner(Matrix{Float64})
    evsObs.phononPosition=FullBinner(Matrix{Float64})
    evsObs.phononMomenta=FullBinner(Matrix{Float64})
    evsObs.energySeries=Vector{Float64}()

    return(evsObs)
end

function getEvEnergy(evs,gens::Generators)::Float64
    energy = 0.0
    d=size(gens.spinOperators[1])[1]
    Id=Matrix((1.0+0im)I,d,d)
    for site in 1:length(evs.latticePrev)
        # get vector of exp values for site
        s0 = getExpValSpin(evs, site)
        p0 = getPhonon(evs.latticePrev, site)
        # tempS0=copy(s0)
        
        # push!(tempS0,calcInnerProd(getSpin(lattice,site),Id,getSpin(lattice,site)))
    

        #two-spin interactions
        interactionSites = getInteractionSites(evs.latticePrev, site)
        interactionMatrices = getInteractionMatrices(evs.latticePrev, site)
        for i in 1:length(interactionSites)
            # get vector of exp values for interaction site
            s1 = getExpValSpin(evs, i)
            # tempS1=copy(s1)
            # push!(tempS1,calcInnerProd(getSpin(lattice,interactionSites[i]),Id,getSpin(lattice,interactionSites[i])))
            if site > interactionSites[i]
                energy += exchangeEnergy(s0, interactionMatrices[i], s1)
            end
        end

        energy += phononPotentialEnergy(evs.latticePrev, p0)
        energy += spinPhononCoupling(evs.latticePrev, s0, p0)
        energy += (evs.phononMomentaPrev[1,site]^2)/(2*evs.phononMass[1])
        energy += (evs.phononMomentaPrev[2,site]^2)/(2*evs.phononMass[2])

        #onsite interaction
       # energy += exchangeEnergy(s0, getInteractionOnsite(lattice, site), s0)

        #field interaction
        #energy += dot(s0, getInteractionField(lattice, site))
    end

    return energy
end


function measureEvObservables!(evs, energy)
    push!(evs.obs.spinStates, evs.lattice.expVals)
    push!(evs.obs.phononPosition, evs.lattice.phonons)
    push!(evs.obs.phononMomenta, evs.phononMomenta)
    push!(evs.obs.energySeries, energy)
end